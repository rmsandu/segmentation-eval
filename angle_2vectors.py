# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 16:14:36 2018

@author: Raluca Sandu
"""

import numpy as np
from math import degrees

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(epNeedle1,tpNeedle1,epNeedle2,tpNeedle2):
    """ Returns the angle in degrees between entry and reference trajectory needles::
            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793        
    """
#        vector1 = TargetPoint-EntryPoint
#        vector2 = TargetPoint-EntryPoint

#    try:
#         epNeedle1 = np.array([float(i) for i in epNeedle1.split()])
#         tpNeedle1 = np.array([float(i) for i in tpNeedle1.split()])
#        
#         epNeedle2 = np.array([float(i) for i in epNeedle2.split()])
#         tpNeedle2 = np.array([float(i) for i in tpNeedle2.split()])
#    except Exception:
        
    if tpNeedle1 is None or epNeedle1 is None or tpNeedle2  is None or epNeedle2 is None:
        return np.nan
        
    v1 = np.array(tpNeedle1) - np.array(epNeedle1)
    v2 = np.array(tpNeedle2) - np.array(epNeedle2)
    
    # angle = atan2(vector2.y, vector2.x) - atan2(vector1.y, vector1.x);
    # if (angle < 0) angle += 2 * M_PI;
        
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    
    # return the value of the angle in degrees 
    angle_radians = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    
    return degrees(angle_radians)
