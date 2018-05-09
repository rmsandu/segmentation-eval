# -*- coding: utf-8 -*-
"""
Created on Tue May  8 13:36:39 2018

@author: Raluca Sandu
"""

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#def AnimateDICOM(SourceImg, MaskImg):
slices, x , y = SourceImg.shape
fig = plt.figure()

im = plt.imshow(SourceImg[0,:,:], cmap=plt.cm.gray, interpolation='none', animated=True)

def updatefig(z):
    global SourceImg
    new_slice = SourceImg[z,:,:]
    im.set_array(new_slice)
    return im

ani = animation.FuncAnimation(fig, updatefig, np.arange(1, slices), interval=20, blit=True)
plt.show() 

