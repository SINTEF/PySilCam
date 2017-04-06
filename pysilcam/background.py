# -*- coding: utf-8 -*-
'''
Moving background correction

use the backgrounder function!

acquire() must produce a float64 np array
'''
import numpy as np


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
#    imbg = np.amax(bgstack, axis=0)  # average the images in the stack
    
    return bgstack, imbg


def shift_bgstack(bgstack, imbg, imnew):
    '''shofts the background by popping the oldest and added a new image
    returns:
      bgstack (updated list of all background images)
      imbg (updated actual background image)
    '''

    # recalculate av_window (so it doesn't need passing)
    av_window = np.shape(bgstack)
    av_window = av_window[0]
    
    imold = bgstack.pop(0)  # pop the oldest image from the stack,
    # but keep it for subtraction from imbg later

    bgstack.append(imnew)  # append the new image to the stack
    
#    imbg = imbg * av_window  # rescale imbg
#    imbg -= imold  # subtract oldest image
#    imbg += imnew  # add new image
#    imbg /= av_window  # convert back to proper average
    imbg = np.mean(bgstack, axis=0)
    
    return bgstack, imbg


def correct_im(imbg, imraw):
    '''corrects raw image by subtracting the background
    inputs:
      imbg (the actual background averaged image)
      imraw (a raw image)

    returns:
      imc (a corrected image)
    '''
    imc = imraw - imbg
    
    m = imc.max()
    imc += 255-m
    imc[imc<0] = 0
    imc[imc>255] = 255
    
    return imc


def shift_and_correct(bgstack, imbg, imraw):
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
      imc (corrcted image)
    '''

    imc = correct_im(imbg, imraw)
    bgstack, imbg = shift_bgstack(bgstack, imbg, imraw)
    
    return bgstack, imbg, imc


def backgrounder(av_window, acquire):
    '''generator which interracts with acquire to return a corrcted image
    given av_window number of frame to use in creating a moving background

    example useage:
      avwind = 10 # number of images used for background
      imgen = backgrounder(avwind,acquire) # setup generator

      n = 10 # acquire 10 images and correct them with a sliding background:
      for i in range(n):
          imc = next(imgen)
          print(i)
    '''

    # Set up initial background image stack
    bgstack, imbg = ini_background(av_window, acquire)

    # Aquire images, apply background correction and yield result
    for timestamp, imraw in acquire:
        bgstack, imbg, imc = shift_and_correct(bgstack, imbg, imraw)
        yield timestamp, imc
