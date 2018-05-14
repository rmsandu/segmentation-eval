# -*- coding: utf-8 -*-
"""
Created on Tue May  8 13:36:39 2018

@author: Raluca Sandu
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#def AnimateDICOM(SourceImg, MaskImg):
#SourceImg = img_source_nda
SourceImg = img_validation_nda
slices, x , y = SourceImg.shape
fig = plt.figure()
plt.grid(False)
im = plt.imshow(SourceImg[0,:,:], cmap=plt.cm.gray, interpolation='none', animated=True)
im2 = plt.imshow(mask_ablation_nda[0,:,:], cmap='RdYlBu', alpha=0.3, interpolation='none',animated=True)
ims = []

def updatefig(z):
    global SourceImg
    global mask_ablation_nda
    
    TumorOverlay = mask_ablation_nda[z,:,:].astype(np.float)
    TumorOverlay[TumorOverlay ==0 ] = np.nan

    new_slice = SourceImg[z,:,:]
    
    im2.set_array(TumorOverlay)
    im.set_array(new_slice)
    ims.append([im])
    ims.append([im2])
    return [ims]

ani = animation.FuncAnimation(fig, updatefig, frames = np.arange(1, slices), interval=20)
# blit =  True option to re-draw only the parts that have changed
# repeat_delay=1000
plt.show() 
# saving doeasn't currently work 14.05.2018
#ani.save('animationTumor.gif', writer='imagemagick', fps=10)
#ani.save('animation.mp4', writer='ffmpeg' ,fps=10)