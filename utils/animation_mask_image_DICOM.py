# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import DicomReader as Reader
import SimpleITK as sitk

#%%

def plot_histogram(SourceImg):
    slices, x, y = SourceImg.shape
    imgs = np.stack([SourceImg[z, :, :] for z in range(slices)])
    fig, ax = plt.subplots()
    col_height1, bins1, patches = ax.hist(imgs.flatten(), bins=50, ec='darkgrey')
    plt.xlabel('Hounsfield Units')
    plt.ylabel('Frequency')

global SourceImg
global tumor_mask_nda
global ablation_mask_nda

df_final = pd.read_excel(
    r"C:\PatientDatasets_GroundTruth_Database\Stockholm\resized\FilepathsResizedGTSegmentations.xlsx")
rootdir_plots = r"C:\PatientDatasets_GroundTruth_Database\Stockholm\plots"
# ablations = df_final['AblationPath']
# tumors = df_final['TumorPath']
ablations = df_final["Ablation Segmentation Path Resized"].tolist()
tumors = df_final["Tumour Segmentation Path Resized"].tolist()
# ablations = df_final["TumorPath"].tolist()
# tumors = df_final["AblationPath"].tolist()
plan = df_final["PlanTumorPath"].tolist()

# def AnimateDICOM(SourceImg, MaskImg):
# SourceImg = img_source_nda
ablation_img_sitk = Reader.read_dcm_series(ablations[0], False)
tumor_img_sitk = Reader.read_dcm_series(tumors[0], False)
source_img_sitk = Reader.read_dcm_series(plan[0], False)

source_img_sitk_cast = sitk.Cast(sitk.RescaleIntensity(source_img_sitk), sitk.sitkUInt16)
tumor_img_sitk_cast = sitk.Cast(sitk.RescaleIntensity(tumor_img_sitk), sitk.sitkUInt16)
ablation_img_sitk_cast = sitk.Cast(sitk.RescaleIntensity(ablation_img_sitk), sitk.sitkUInt16)

SourceImg = sitk.GetArrayFromImage(source_img_sitk_cast)
SourceImg = SourceImg.astype(np.float)
SourceImg[SourceImg < 20] = np.nan
# SourceImg[SourceImg > 720] = np.nan
# SourceImg[SourceImg == 0] = np.nan


tumor_mask_nda = sitk.GetArrayFromImage(tumor_img_sitk_cast)
ablation_mask_nda = sitk.GetArrayFromImage(ablation_img_sitk_cast)

slices, x, y = SourceImg.shape
fig = plt.figure()
plt.grid(False)
im = plt.imshow(SourceImg[0, :, :], cmap=plt.cm.gray, interpolation='none', animated=True)
im2 = plt.imshow(tumor_mask_nda[0, :, :], cmap='RdYlBu', alpha=0.3, interpolation='none', animated=True)
im3 = plt.imshow(ablation_mask_nda[0,:,:], cmap='winter', alpha=0.3, interpolation='none', animated=True)
ims = []


def updatefig(z):

    TumorOverlay = tumor_mask_nda[z, :, :].astype(np.float)
    TumorOverlay[TumorOverlay == 0] = np.nan

    AblationOverlay = ablation_mask_nda[z,:,:].astype(np.float)
    AblationOverlay[AblationOverlay == 0] = np.nan

    new_slice = SourceImg[z, :, :]

    im2.set_array(TumorOverlay)
    im3.set_array(AblationOverlay)
    im.set_array(new_slice)
    ims.append([im])
    ims.append([im2])
    ims.append([im3])
    return [ims]


ani = animation.FuncAnimation(fig, updatefig, frames=np.arange(1, slices), interval=10)

# plot_histogram(SourceImg)
# blit =  True option to re-draw only the parts that have changed
# repeat_delay=1000
plt.show()
# saving doeasn't currently work 14.05.2018
# ani.save('animationTumor.gif', writer='imagemagick', fps=10)
# ani.save('animation.mp4', writer='ffmpeg' ,fps=10)