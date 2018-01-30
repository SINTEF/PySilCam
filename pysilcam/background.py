# -*- coding: utf-8 -*-
'''
Moving background correction

use the backgrounder function!

acquire() must produce a float64 np array
'''
import numpy as np
import logging

#Get module-level logger
logger = logging.getLogger(__name__)

def ini_background(av_window, acquire):
    '''av_window is the number of images to use in creating the background
    function returns:
      bgstack (list of all background iamges)
      imbg (the actual background image)
    '''
    bgstack = []
    bgstack.append(next(acquire)[1])  # get the first image

    for i in range(av_window-1):  # loop through the rest, appending to bgstack
        bgstack.append(next(acquire)[1])

    imbg = np.mean(bgstack, axis=0)  # average the images in the stack

    return bgstack, imbg


def shift_bgstack_accurate(bgstack, imbg, imnew, stacklength):
    '''shifts the background by popping the oldest and added a new image
    returns:
    bgstack (updated list of all background images)
    imbg (updated actual background image)
    '''
    imold = bgstack.pop(0)  # pop the oldest image from the stack,
    bgstack.append(imnew)  # append the new image to the stack
    imbg = np.mean(bgstack, axis=0)
    return bgstack, imbg


def shift_bgstack_fast(bgstack, imbg, imnew, stacklength):
    '''shifts the background by popping the oldest and added a new image
    returns:
      bgstack (updated list of all background images)
      imbg (updated actual background image)
    '''
    imold = bgstack.pop(0)  # pop the oldest image from the stack,
    # subtract the old image from the average (scaled by the average window)
    imbg -= (imold/stacklength)
    # add the new image to the average (scaled by the average window)
    imbg += (imnew/stacklength)
    bgstack.append(imnew)  # append the new image to the stack
    return bgstack, imbg


def correct_im_accurate(imbg, imraw):
    '''corrects raw image by subtracting the background
    inputs:
      imbg (the actual background averaged image)
      imraw (a raw image)

    returns:
      imc (a corrected image)
    '''

    imc = np.float64(imraw) - np.float64(imbg)
    imc[:,:,0] += (255/2 - np.percentile(imc[:,:,0], 50))
    imc[:,:,1] += (255/2 - np.percentile(imc[:,:,1], 50))
    imc[:,:,2] += (255/2 - np.percentile(imc[:,:,2], 50))
    #imc += 255 - np.percentile(imc, 99)
    imc += 255 - imc.max()

    imc[imc>255] = 255
    imc[imc<0] = 0
    imc = np.uint8(imc)

    return imc


def correct_im_fast(imbg, imraw):
    '''corrects raw image by subtracting the background
    inputs:
      imbg (the actual background averaged image)
      imraw (a raw image)

    returns:
      imc (a corrected image)
    '''
    imc = imraw - imbg

    imc += 215
    imc[imc<0] = 0
    imc[imc>255] = 255
    imc = np.uint8(imc)

    return imc


def shift_and_correct(bgstack, imbg, imraw, stacklength, real_time_stats=False):
    '''shifts the background stack and averaged image and corrects the new
    raw image.

    This is a wrapper for shift_bgstack and correct_im

    inputs:
      bgstack (old background stack)
      imbg (old averaged image)
      imraw (a new raw image)

    returns:
      bgstack (updated stack)
      imbg (updated averaged image)
      imc (corrected image)
    '''

    if real_time_stats:
        imc = correct_im_fast(imbg, imraw)
        bgstack, imbg = shift_bgstack_fast(bgstack, imbg, imraw, stacklength)
    else:
        imc = correct_im_accurate(imbg, imraw)
        bgstack, imbg = shift_bgstack_accurate(bgstack, imbg, imraw, stacklength)

    return bgstack, imbg, imc


def backgrounder(av_window, acquire, bad_lighting_limit=None,
        real_time_stats=False):
    '''generator which interacts with acquire to return a corrected image
    given av_window number of frame to use in creating a moving background

    example useage:
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

        if not (bad_lighting_limit==None):
            bgstack_new, imbg_new, imc = shift_and_correct(bgstack, imbg,
                    imraw, stacklength, real_time_stats)

            # basic check of image quality
            r = imc[:, :, 0]
            g = imc[:, :, 1]
            b = imc[:, :, 2]
            s = np.std([r, g, b])
            # ignore bad images
            if (s <= bad_lighting_limit):
                bgstack = bgstack_new
                imbg = imbg_new
                yield timestamp, imc, imraw
            else:
                logger.info('bad lighting, std={0}'.format(s))
        else:
            bgstack, imbg, imc = shift_and_correct(bgstack, imbg, imraw,
                    stacklength)
            yield timestamp, imc, imraw
