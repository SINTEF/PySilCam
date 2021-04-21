# -*- coding: utf-8 -*-
'''
Moving background correction

use the backgrounder function!

acquire() must produce a float64 np array
'''
import numpy as np
import logging
from pysilcam.acquisition import addToQueue

import os
import matplotlib.pyplot as plt

# Get module-level logger
logger = logging.getLogger(__name__)


class Backgrounder():
    '''
    Class used to run background collection via .run() function.
    '''
    def __init__(self, av_window, bad_lighting_limit=None, real_time_stats=False):
        self.av_window = av_window
        self.bad_lighting_limit = bad_lighting_limit
        self.real_time_stats = real_time_stats
        self.bgstack = []
        self.bgstacklength = None
        self.imbg = None

    def ini_background(self, raw_image_queue):
        '''
        Create and initial background stack and average image

        Args:
        Returns:
            bgstack (list)              : list of all images in the background stack
            imbg (uint8)                : background image
        '''
        print('ini_background(self, raw_image_queue)')
        if self.bgstack:
            print("bgstack not empty, resetting")
            self.bgstack = []
        for i in range(self.av_window):  # loop through the rest, appending to bgstack
            print(raw_image_queue.get()[0])
            self.bgstack.append(raw_image_queue.get()[1])

        self.bgstacklength = len(self.bgstack)
        print('self.bgstacklength:', self.bgstacklength)
        self.imbg = np.mean(self.bgstack, axis=0)  # average the images in the stack

    def correct_im_accurate(self, imraw):
        '''
        Corrects raw image by subtracting the background and scaling the output

        There is a small chance of clipping of imc in both crushed blacks an blown
        highlights if the background or raw images are very poorly obtained

        Args:
        imbg (uint8)  : background averaged image
        imraw (uint8) : raw image

        Returns:
        imc (uint8)   : corrected image
        '''

        imc = np.float64(imraw) - np.float64(self.imbg)
        imc[:, :, 0] += (255 / 2 - np.percentile(imc[:, :, 0], 50))
        imc[:, :, 1] += (255 / 2 - np.percentile(imc[:, :, 1], 50))
        imc[:, :, 2] += (255 / 2 - np.percentile(imc[:, :, 2], 50))
        imc += 255 - imc.max()

        imc[imc > 255] = 255
        imc[imc < 0] = 0
        imc = np.uint8(imc)

        return imc

    def correct_im_fast(self, imraw):
        '''
        Corrects raw image by subtracting the background and clipping the ouput
        without scaling

        There is high potential for clipping of imc in both crushed blacks an blown
        highlights, especially if the background or raw images are not properly obtained

        Args:
        imbg (uint8)  : background averaged image
        imraw (uint8) : raw image

        Returns:
        imc (uint8)   : corrected image
        '''
        imc = imraw - self.imbg

        imc += 215
        imc[imc < 0] = 0
        imc[imc > 255] = 255
        imc = np.uint8(imc)

        return imc

    def shift_bgstack_accurate(self, imnew, inplace=True):
        '''
        Shifts the background by popping the oldest and added a new image. self.bgstack
        and self.imbg are then updated.

        The new background is calculated slowly by computing the mean of all images
        in the background stack.

        Args:
            imnew (unit8)       : new image to be added to stack
            inplace (bool)      : if to update self.bgstack and self.imbg inplace or
                                  return new (updated) objects for these
        '''

        if inplace:
            _ = self.bgstack.pop(0)  # pop the oldest image from the stack,
            self.bgstack.append(imnew)  # append the new image to the stack
            self.imbg = np.mean(self.bgstack, axis=0)
            return None
        else:
            bgstack_new = self.bgstack.copy()
            _ = bgstack_new.pop(0)
            bgstack_new.append(imnew)
            imbg_new = np.mean(bgstack_new, axis=0)
            return bgstack_new, imbg_new

    def shift_bgstack_fast(self, imnew, inplace=True):
        '''
        Shifts the background by popping the oldest and added a new image. self.bgstack
        and self.imbg are then updated.

        The new background is appoximated quickly by subtracting the old image and
        adding the new image (both scaled by the bgstacklength).
        This is close to a running mean, but not quite.

        Args:
            imnew (unit8)       : new image to be added to stack
            inplace (bool)      : if to update self.bgstack and self.imbg inplace or
                                  return new (updated) objects for these
        '''
        if inplace:
            imold = self.bgstack.pop(0)  # pop the oldest image from the stack,
            # subtract the old image from the average (scaled by the average window)
            self.imbg -= (imold / self.bgstacklength)
            # add the new image to the average (scaled by the average window)
            self.imbg += (imnew / self.bgstacklength)
            self.bgstack.append(imnew)  # append the new image to the stack
            return None
        else:
            bgstack_new = self.bgstack.copy()
            imbg_new = self.imbg.copy()
            imold = bgstack_new.pop(0)
            imbg_new -= (imold / self.bgstacklength)
            imbg_new += (imnew / self.bgstacklength)
            bgstack_new.append(imnew)
            return bgstack_new, imbg_new

    def shift_and_correct(self, imraw, inplace=True):
        '''
        Shifts the background stack and averaged image and corrects the new
        raw image.

        This is a wrapper for shift_bgstack and correct_im

        Args:
            imraw (uint8)                   : raw image

        Returns:
            imc (uint8)                     : corrected image
        '''

        if self.real_time_stats:
            imc = self.correct_im_fast(imraw)
            stack_data = self.shift_bgstack_fast(imraw, inplace=inplace)
        else:
            imc = self.correct_im_accurate(imraw)
            stack_data = self.shift_bgstack_accurate(imraw, inplace=inplace)

        if inplace:
            return imc
        else:
            return imc, stack_data[0], stack_data[1]

    def run(self, config_filename, raw_image_queue, proc_image_queue):

        print("self.ini_background(raw_image_queue)")
        # Set up initial background image stack
        self.ini_background(raw_image_queue)
        print("self.ini_background(raw_image_queue) - OK")

        while True:
            # get raw image from queue
            timestamp, imraw = raw_image_queue.get()
            print(timestamp, "Image being processed")

            if self.bad_lighting_limit is not None:
                imc, bgstack_new, imbg_new = self.shift_and_correct(imraw, inplace=False)

                # basic check of image quality
                r = imc[:, :, 0]
                g = imc[:, :, 1]
                b = imc[:, :, 2]
                s = np.std([r, g, b])
                # ignore bad images
                if s <= self.bad_lighting_limit:
                    self.bgstack = bgstack_new
                    self.imbg = imbg_new

                    if proc_image_queue is not None:
                        logger.debug('Adding image to processing queue: ' + str(timestamp))
                        print('Adding image to processing queue: ' + str(timestamp))
                        addToQueue(self.real_time_stats, proc_image_queue, 0, timestamp, imc)  # the tuple (i, timestamp, imc) is added to the inputQueue
                        logger.debug('Processing queue updated')
                        print('Processing queue updated')

                else:
                    logger.info('bad lighting, std={0}'.format(s))
                    print("bad lighting!!!")
            else:
                print("No bad lighting limit. Just standard stuff.")
                imc = self.shift_and_correct(imraw)
                # plt.imshow(imc)
                # plt.show()
                # input()
                # filename = os.path.join("pyvimba_test", timestamp.strftime('D%Y%m%dT%H%M%S.%f.silc'))
                # with open(filename, 'wb') as fh:
                #     np.save(fh, imc, allow_pickle=False)
                #     fh.flush()
                #     os.fsync(fh.fileno())
                if proc_image_queue is not None:
                    logger.debug('Adding image to processing queue: ' + str(timestamp))
                    addToQueue(self.real_time_stats, proc_image_queue, 0, timestamp, imc)  # the tuple (i, timestamp, imc) is added to the inputQueue
                    logger.debug('Processing queue updated')

            # correct with background stack

            # update background stack

            # for testing either:
            # plt.imshow(im_corrected)
            # input()
            # or:
            # write the corrected image to disc
            # if we get here, we are happy.

            # (add corrected images to a queue for processing)


def backgrounder_OLD_VERSION(av_window, acquire, bad_lighting_limit=None,
                 real_time_stats=False):
    '''
    Generator which interacts with acquire to return a corrected image
    given av_window number of frame to use in creating a moving background

    Args:
        av_window (int)               : number of images to use in creating the background
        acquire (generator object)    : acquire generator object created by the Acquire class
        bad_lighting_limit=None (int) : if a number is supplied it is used for throwing away raw images that have a
                                        standard deviation in colour which exceeds the given value

    Yields:
        timestamp (timestamp)         : timestamp of when raw image was acquired 
        imc (uint8)                   : corrected image ready for analysis or plotting
        imraw (uint8)                 : raw image

    Useage:
      avwind = 10 # number of images used for background
      imgen = backgrounder(avwind,acquire,bad_lighting_limit) # setup generator

      n = 10 # acquire 10 images and correct them with a sliding background:
      for i in range(n):
          imc = next(imgen)
          print(i)
    '''

    # Set up initial background image stack
    bgstack, imbg = ini_background(av_window, acquire)
    stacklength = len(bgstack)

    # Aquire images, apply background correction and yield result
    for timestamp, imraw in acquire:

        if bad_lighting_limit is not None:
            bgstack_new, imbg_new, imc = shift_and_correct(bgstack, imbg,
                                                           imraw, stacklength, real_time_stats)

            # basic check of image quality
            r = imc[:, :, 0]
            g = imc[:, :, 1]
            b = imc[:, :, 2]
            s = np.std([r, g, b])
            # ignore bad images
            if s <= bad_lighting_limit:
                bgstack = bgstack_new
                imbg = imbg_new
                yield timestamp, imc, imraw
            else:
                logger.info('bad lighting, std={0}'.format(s))
        else:
            bgstack, imbg, imc = shift_and_correct(bgstack, imbg, imraw,
                                                   stacklength, real_time_stats)
            yield timestamp, imc, imraw
