# -*- coding: utf-8 -*-

import numpy as np

'''aquire() must produce a float64 np array
EXAMPLE:

imgen = backgrounder(10)

for i in range(10):
    imc = next(imgen)
    print(i)

'''


def aquire():
    im = np.zeros((2048,2448),dtype=np.float64)
    return im

def ini_background(av_window):
    
    bgstack = []
    bgstack.append(aquire())
    
    for i in range(av_window-1):
        bgstack.append(aquire())
    
    imbg = np.mean(bgstack,0)
    
    return bgstack, imbg

def shift_bgstack(bgstack,imbg,imnew):
    av_window = np.shape(bgstack)
    av_window = av_window[0]
    
    imold = bgstack.pop(0)
    bgstack.append(imnew)
    
    imbg = imbg * av_window
    imbg -= imold
    imbg += imnew
    imbg /= av_window
    
    return bgstack, imbg

def correct_im(imbg,imraw):
    imc = imbg - imraw
    
    m = imc.max()
    imc += 255-m
    
    return imc

def shift_and_correct(bgstack,imbg,imraw):
    bgstack, imbg = shift_bgstack(bgstack,imbg,imraw)
    imc = correct_im(imbg,imraw)
    return bgstack, imbg, imc

def backgrounder(av_window):
    bgstack, imbg = ini_background(10)
    
    while True:
        imraw = aquire()
        bgstack, imbg, imc = shift_and_correct(bgstack,imbg,imraw)
        yield imc
