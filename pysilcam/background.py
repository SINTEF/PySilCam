# -*- coding: utf-8 -*-
'''
Moving background correction

use the backgrounder function!

acquire() must produce a float64 np array
'''
import numpy as np
import logging
import matplotlib.pyplot as plt

# Get module-level logger
logger = logging.getLogger(__name__)


def ini_background(av_window, acquire):
    '''
    Create and initial background stack and average image
    
    Args:
        av_window (int)             : number of images to use in creating the background
        acquire (generator object)  : acquire generator object created by the Acquire class
    Returns:
        bgstack (list)              : list of all images in the background stack
        imbg (uint8)                : background image
    '''
    bgstack = []
    bgstack.append(next(acquire)[1])  # get the first image

    for i in range(av_window - 1):  # loop through the rest, appending to bgstack
        bgstack.append(next(acquire)[1])

    imbg = np.mean(bgstack, axis=0)  # average the images in the stack

    return bgstack, imbg


def shift_bgstack_accurate(bgstack, imbg, imnew, stacklength):
    '''
    Shifts the background by popping the oldest and added a new image
    
    The new background is calculated slowly by computing the mean of all images
    in the background stack.
    
    Args:
        bgstack (list)      : list of all images in the background stack
        imbg (uint8)        : background image
        imnew (unit8)       : new image to be added to stack
        stacklength (int)   : unsed here - it is just there to maintain the same behaviour as shift_bgstack_fast()
        
    Returns:
        bgstack (updated list of all background images)
        imbg (updated actual background image)
    '''
    imold = bgstack.pop(0)  # pop the oldest image from the stack,
    bgstack.append(imnew)  # append the new image to the stack
    imbg = np.mean(bgstack, axis=0)
    return bgstack, imbg


def shift_bgstack_fast(bgstack, imbg, imnew, stacklength):
    '''
    Shifts the background by popping the oldest and added a new image
    
    The new background is appoximated quickly by subtracting the old image and
    adding the new image (both scaled by the stacklength).
    This is close to a running mean, but not quite.
    
    Args:
        bgstack (list)      : list of all images in the background stack
        imbg (uint8)        : background image
        imnew (unit8)       : new image to be added to stack
        stacklength (int)   : unsed int here - just there to maintain the same behaviour as shift_bgstack_fast()
        
    Returns:
        bgstack (updated list of all background images)
        imbg (updated actual background image)
    '''
    imold = bgstack.pop(0)  # pop the oldest image from the stack,
    # subtract the old image from the average (scaled by the average window)
    imbg -= (imold / stacklength)
    # add the new image to the average (scaled by the average window)
    imbg += (imnew / stacklength)
    bgstack.append(imnew)  # append the new image to the stack
    return bgstack, imbg


def correct_im_accurate(imbg, imraw):
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

    imc = np.float64(imraw) - np.float64(imbg)
    imc[:, :, 0] += (255 / 2 - np.percentile(imc[:, :, 0], 50))
    imc[:, :, 1] += (255 / 2 - np.percentile(imc[:, :, 1], 50))
    imc[:, :, 2] += (255 / 2 - np.percentile(imc[:, :, 2], 50))
    imc += 255 - imc.max()

    imc[imc > 255] = 255
    imc[imc < 0] = 0
    imc = np.uint8(imc)

    return imc


def correct_im_fast(imbg, imraw):
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
    imc = imraw - imbg

    imc += 215
    imc[imc < 0] = 0
    imc[imc > 255] = 255
    imc = np.uint8(imc)

    return imc


def shift_and_correct(bgstack, imbg, imraw, stacklength, real_time_stats=False):
    '''
    Shifts the background stack and averaged image and corrects the new
    raw image.

    This is a wrapper for shift_bgstack and correct_im      
      
    Args:
        bgstack (list)                  : list of all images in the background stack
        imbg (uint8)                    : background image
        imraw (uint8)                   : raw image
        stacklength (int)               : unsed int here - just there to maintain the same behaviour as
                                          shift_bgstack_fast()
        real_time_stats=False (Bool)    : if True use fast functions, if False use accurate functions
        
    Returns:
        bgstack (list)                  : list of all images in the background stack
        imbg (uint8)                    : background averaged image
        imc (uint8)                     : corrected image
    '''

    if real_time_stats:
        imc = correct_im_fast(imbg, imraw)
        bgstack, imbg = shift_bgstack_fast(bgstack, imbg, imraw, stacklength)
    else:
        imc = correct_im_accurate(imbg, imraw)
        bgstack, imbg = shift_bgstack_accurate(bgstack, imbg, imraw, stacklength)

    return bgstack, imbg, imc


class Backgrounder():
    '''
    Class used to run background collection via .run() function.
    '''
    def __init__(self, av_window, bad_lighting_limit=None, real_time_stats=False):
        self.av_window = av_window
        self.bad_lighting_limit = bad_lighting_limit,
        self.real_time_stats = real_time_stats
        # These can wait to check that the fns work first.
        # self.bgstack = []
        # self.bgstacklength = None
        # self.imbg = None

    def ini_background(self, raw_image_queue):
        '''
        Create and initial background stack and average image

        Args:
        Returns:
            bgstack (list)              : list of all images in the background stack
            imbg (uint8)                : background image
        '''
        bgstack = []
        for i in range(self.av_window):  # loop through the rest, appending to bgstack
            print(raw_image_queue.get()[0])
            bgstack.append(raw_image_queue.get()[1])

        imbg = np.mean(bgstack, axis=0)  # average the images in the stack

        return bgstack, imbg

    def run(self, config_filename, raw_image_queue):

        print("In Backgrounder.run(), woohoo")
        # Set up initial background image stack
        bgstack, imbg = self.ini_background(raw_image_queue)
        stacklength = len(bgstack)

        while True:
            # get raw image from queue
            timestamp, imraw = raw_image_queue.get()
            print(timestamp, imraw)

            if self.bad_lighting_limit is not None:
                bgstack_new, imbg_new, imc = shift_and_correct(
                    bgstack, imbg, imraw, stacklength, self.real_time_stats)

                # basic check of image quality
                r = imc[:, :, 0]
                g = imc[:, :, 1]
                b = imc[:, :, 2]
                s = np.std([r, g, b])
                # ignore bad images
                if s <= self.bad_lighting_limit:
                    bgstack = bgstack_new
                    imbg = imbg_new
                    # yield timestamp, imc, imraw
                else:
                    logger.info('bad lighting, std={0}'.format(s))
                    print("bad lighting!!!")
            else:
                print("No bad lighting limit. Just standard stuff.")
                bgstack, imbg, imc = shift_and_correct(
                    bgstack, imbg, imraw, stacklength, self.real_time_stats)
                plt.imshow(imc)
                input()
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
