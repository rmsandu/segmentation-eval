# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 16:59:22 2018

@author: Raluca Sandu
"""

import pandas as pd
import untangle as ut

filename = "CAS-One MWA_Database.xml"
''' try to open and parse the xml filename
    if error, return message
'''
# TO DO: add the patient ID from the folder name
try:
    xmlobj = ut.parse(filename)
except Exception:
    print('XML file structure is broken, cannot read XML')

dict_mwa_info = []
mwa_needle_info =  xmlobj.Eagles.Database.MWA
for needle in mwa_needle_info:
    # "Acculis MTA" has id=0
    if needle['id']=='0':
        acculis_needle = needle
        ablation_params = acculis_needle.AblationParameters.Geometry.Shape
        for idx, ablation_param in enumerate(ablation_params):
            print(ablation_param)
            mwa_info = {     'NeedleType': "Acculis MTA",
                             'Shape': idx,
                             'Type': ablation_param["type"],
                             'Power': ablation_param["power"],
                             'Time_seconds': ablation_param["time"],
                             'Radii': ablation_param["radii"],
                             'Translation': ablation_param["translation"],
                             'Rotation': ablation_param["rotation"]
                        }
            dict_mwa_info.append(mwa_info)
            
df_mwa = pd.DataFrame(dict_mwa_info)