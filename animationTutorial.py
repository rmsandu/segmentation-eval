# -*- coding: utf-8 -*-
"""
Created on Wed May  9 16:03:16 2018

@author: Raluca Sandu
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig = plt.figure()


def f(x, y):
    return np.sin(x) + np.cos(y)

x = np.linspace(0, 2 * np.pi, 120)
y = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)
# ims is a list of lists, each row is a list of artists to draw in the
# current frame; here we are just animating one artist, the image, in
# each frame
global SourceImg
global mask_tumor_nda
ims = []
for z in range(240):
    x += np.pi / 15.
    y += np.pi / 20.
#    im = plt.imshow(f(x, y), animated=True)
    im = plt.imshow(SourceImg[z,:,:], cmap=plt.cm.gray, interpolation='none', animated=True)
    ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=20, blit=True,
                                repeat_delay=1000)

# ani.save('dynamic_images.mp4')

plt.show()