# -*- coding: utf-8 -*-
"""
Created on Mon May  7 16:27:36 2018

@author: Raluca Sandu
"""

import SimpleITK as sitk
import matplotlib.pyplot as plt

def sitk_show(imgSitk,  title=None, margin=0.05, dpi=40 ):

    
    nda = sitk.GetArrayFromImage(imgSitk)
    spacing = imgSitk.GetSpacing()
    figsize = (1 + margin) * nda.shape[0] / dpi, (1 + margin) * nda.shape[1] / dpi
    extent = (0, nda.shape[1]*spacing[1], nda.shape[0]*spacing[0], 0)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])

    plt.set_cmap("gray")
    ax.imshow(nda,extent=extent,interpolation=None)
    
    if title:
        plt.title(title)
    
    plt.show()