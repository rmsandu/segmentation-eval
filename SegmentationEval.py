
# coding: utf-8

# # Segmentation Evaluation In Python - metrics for volume overlap

# Raluca Sandu November 2017

#%% Libraries Import
import os
import numpy as np
import pandas as pd
from enum import Enum
from medpy import metric
import SimpleITK as sitk
from scipy import ndimage
from surface import Surface
import graphing as gh
import matplotlib.pyplot as plt
#plt.style.use('ggplot')
#plt.style.use('seaborn')
#print(plt.style.available)


#%%
''' Volume Overlap measures:
    - Dice (itk)
    - Jaccard (itk)
    - Volumetric Similarity (itk)
    - Volumetric Overlap Error (voe) - @package medpy.metric.surface by Oskar Maier. calls function "surface.py"
    - Relative Volume Difference (rvd) - Volumetric Overlap Error (voe) - @package medpy.metric.surface by Oskar Maier. calls function "surface.py"

'''
''' Surface Distance Measures:
     - Maximum Surface Distance (which might be hausdorff need to check) - itk
     - Hausdorff Distance - itk
     - Mean Surface Distance - itk
     - Median Surface Distance - itk
     - rmsd : root mean square symmetric surface distance [mm] -  @package medpy.metric.surface by Oskar Maier. calls function "surface.py"
     - assd:  average symmetric surface distance [mm] - -  @package medpy.metric.surface by Oskar Maier. calls function "surface.py"
'''
# Use enumerations to represent the various evaluation measures
# very stupid way to do it atm, change it later
class OverlapMeasures(Enum):
     dice, jaccard, volume_similarity, volumetric_overlap_error, relative_vol_difference = range(5)

class SurfaceDistanceMeasures(Enum):
    hausdorff_distance,max_surface_distance, mean_symmetric_surface_distance, median_symmetric_surface_distance, std_deviation = range(5)

#%% 
'''Read Segmentations and their Respective Ablation Zone
    Assumes both maks are isotropic (have the same number of slices).
    Foreground value label = 255 [white] and is considered to be the object of interest.
    Background object = 0
    
    --- This function requieres a root directory filepath that contains segmnations of ablations and tumors.
        it assumes the parent directory is named after the patient name/id
        Input : rootdirectory filepath
        Output: dataframe containing filepaths of segmentations and ablations for a specific patient.    
'''

segmentation_data = [] # list of dictionaries containing the filepaths of the segmentations

rootdir = "C:/Users/Raluca Sandu/Documents/LiverInterventionsBern_Ablations/studyPatientsMasks/"

for subdir, dirs, files in os.walk(rootdir):
    tumorFilePath  = ''
    ablationSegm = ''
    for file in files:
        if file == "tumorSegm":
            FilePathName = os.path.join(subdir, file)
            tumorFilePath = os.path.normpath(FilePathName)
        elif file == "ablationSegm":
            FilePathName = os.path.join(subdir, file)
            ablationFilePath = os.path.normpath(FilePathName)
        else:
            print("")
        
    if (tumorFilePath) and (ablationFilePath):
        dir_name = os.path.dirname(ablationFilePath)
        dirname2 = os.path.split(dir_name)[1]
        data = {'PatientName' : dirname2,
                'TumorFile' : tumorFilePath,
                'AblationFile' : ablationFilePath
                }
        segmentation_data.append(data)  

# convert to data frame 
df_patientdata = pd.DataFrame(segmentation_data)
df_metrics = pd.DataFrame() # emtpy dataframe to append the segmentation metrics calculated

ablations = df_patientdata['AblationFile'].tolist()
segmentations = df_patientdata['TumorFile'].tolist()
pats = df_patientdata['PatientName']
#%%

for idx, seg in enumerate(segmentations):
    image = sitk.ReadImage(seg, sitk.sitkUInt8) # reference images
    segmentation = sitk.ReadImage(ablations[idx],sitk.sitkUInt8) # segmentations to compare to the GT segmentations

    '''init vectors (size) that will contain the volume and distance metrics'''
    '''init the OverlapMeasures Image Filter and the HausdorffDistance Image Filter from Simple ITK'''

    overlap_results = np.zeros((1,len(OverlapMeasures.__members__.items())))  
    surface_distance_results = np.zeros((1,len(SurfaceDistanceMeasures.__members__.items())))  
    
    overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
    hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()
    
    reference_segmentation = image # the refenrence image in this case is the tumor mask
    label = 255
    # init signed mauerer distance as reference metrics
    reference_distance_map = sitk.SignedMaurerDistanceMap(reference_segmentation, squaredDistance=False, useImageSpacing=True)
    label_intensity_statistics_filter = sitk.LabelIntensityStatisticsImageFilter()
 
    ''' Calculate Overlap Metrics'''
    volscores = {}
    ref = sitk.GetArrayFromImage(image) # convert the sitk image to numpy array 
    seg = sitk.GetArrayFromImage(segmentation)
    volscores['rvd'] = metric.ravd(seg, ref)
    volscores['dice'] = metric.dc(seg,ref)
    volscores['jaccard'] = metric.binary.jc(seg,ref)
    volscores['voe'] = 1. - volscores['jaccard']
   

    ''' Add the Volume Overlap Metrics in the Enum vector '''
    overlap_measures_filter.Execute(reference_segmentation, segmentation)
    overlap_results[0,OverlapMeasures.jaccard.value] = overlap_measures_filter.GetJaccardCoefficient()
    overlap_results[0,OverlapMeasures.dice.value] = overlap_measures_filter.GetDiceCoefficient()
    overlap_results[0,OverlapMeasures.volume_similarity.value] = overlap_measures_filter.GetVolumeSimilarity()
    overlap_results[0,OverlapMeasures.volumetric_overlap_error.value] = volscores['voe']
    overlap_results[0,OverlapMeasures.relative_vol_difference.value] = volscores['rvd'] 
    ''' Add the Surface Distance Metrics in the Enum vector '''
 
    # Surface distance measures
    segmented_surface = sitk.LabelContour(segmentation)

    
    label_intensity_statistics_filter.Execute(segmented_surface, reference_distance_map)

      # Hausdorff distance
    hausdorff_distance_filter.Execute(reference_segmentation, segmentation)
    surface_distance_results[0,SurfaceDistanceMeasures.hausdorff_distance.value] = hausdorff_distance_filter.GetHausdorffDistance()
    surface_distance_results[0,SurfaceDistanceMeasures.mean_symmetric_surface_distance.value] = label_intensity_statistics_filter.GetMean(label)
    surface_distance_results[0,SurfaceDistanceMeasures.median_symmetric_surface_distance.value] = label_intensity_statistics_filter.GetMedian(label)
    surface_distance_results[0,SurfaceDistanceMeasures.std_deviation.value] = label_intensity_statistics_filter.GetStandardDeviation(label)
    surface_distance_results[0,SurfaceDistanceMeasures.max_surface_distance.value] = label_intensity_statistics_filter.GetMaximum(label)

    ''' Graft our results matrix into pandas data frames '''
    overlap_results_df = pd.DataFrame(data=overlap_results, index = list(range(1)), 
                                      columns=[name for name, _ in OverlapMeasures.__members__.items()])
     
    surface_distance_results_df = pd.DataFrame(data=surface_distance_results, index = list(range(1)), 
                                      columns=[name for name, _ in SurfaceDistanceMeasures.__members__.items()]) 
    
  
    #change DataFrame column names 
    overlap_results_df.columns = ['Dice', 'Jaccard', 'Volume Similarity', 'Volume Overlap Error', 'Relative Volume Difference']
    surface_distance_results_df.columns = [ 'Hausdorff Distance', 'Maximum Surface Distance', 'Average Distance ', 'Median Distance', 'Standard Deviation']
    metrics_all = pd.concat([overlap_results_df, surface_distance_results_df], axis=1)
    df_metrics = df_metrics.append(metrics_all)
    df_metrics.index = list(range(len(df_metrics)))

    #%% 
    '''edit axis limits & labels '''
    ''' save plots'''
#    figName_vol = pats[idx] + 'volumeMetrics'
#    figpath1 = os.path.join(rootdir, figName_vol)
#    fig, ax = plt.subplots()
#    dfVolT = overlap_results_df.T
##    plt.style.use('ggplot')
#    color = plt.cm.Dark2(np.arange(len(dfVolT))) # create colormap
##    color=color
#    dfVolT.plot(kind='bar', rot=15, legend=False, ax=ax, grid=True, color='coral')
#    plt.axhline(0, color='k')
#    plt.ylim((-1.5,1.5))
#    plt.tick_params(labelsize=12)
#    plt.title('Volumetric Overlap Metrics. Ablation GT vs. Ablation Estimated. Patient ' + str(idx+1))
#    plt.rc('figure', titlesize=25) 
#    gh.save(figpath1,width=12, height=10)
#    
#    # PLOT SURFACE DISTANCE METRICS
#    figName_distance = pats[idx] + 'distanceMetrics'
#    figpath2 = os.path.join(rootdir, figName_distance)
##    plt.style.use('seaborn-colorblind')
#    dfDistT = surface_distance_results_df.T
#    color = plt.cm.Dark2(np.arange(len( dfDistT))) # create colormap
##    color=color
#    fig1, ax1 = plt.subplots()
#    dfDistT.plot(kind='bar',rot=15, legend=False, ax=ax1, grid=True)
#    plt.ylabel('[mm]')
#    plt.axhline(0, color='k')
#    plt.ylim((0,30))
#    plt.title('Surface Distance Metrics. Ablation GT vs. Ablation Estimated. Patient ' + str(idx+1))
#    plt.rc('figure', titlesize=25) 
#    plt.tick_params(labelsize=12)
#    gh.save(figpath2,width=12, height=10)
    
    # PLOT THE HISTOGRAM FOR THE MAUERER DISTANCES
    figName_slice = pats[idx] + 'Slice'
    figpathSlice = os.path.join(rootdir, figName_slice)
    segmented_surface_float = sitk.GetArrayFromImage(segmented_surface)
    reference_distance_map_float = sitk.GetArrayFromImage(reference_distance_map)
    dists_fromAblation = segmented_surface_float * reference_distance_map_float
    
    segmented_surface_ref = sitk.LabelContour(reference_segmentation)
    mask_distance_map = sitk.SignedMaurerDistanceMap(segmentation, squaredDistance=False, useImageSpacing=True)
    
    segmented_surface_ref_float = sitk.GetArrayFromImage(segmented_surface_ref)
    mask_distance_map_float = sitk.GetArrayFromImage(mask_distance_map)
    dists_fromTumor = segmented_surface_ref_float * mask_distance_map_float
#    fig2, ax2= plt.subplots()
#    z = int(np.floor(np.shape(dists_to_plot)[0]/2))
#    plt.imshow(dists_to_plot[z,:,:]/255)
#    plt.title(' Distance Map. 1 Slice Visualization. Patient ' + str(idx+1))
#    plt.rc('figure', titlesize=25) 
#    gh.save(figpathSlice, width=12, height=10)
#    
    figName_hist = pats[idx] + 'histogramDistancesfromAblationtoTumor'
    figpathHist = os.path.join(rootdir, figName_hist)
    ix = dists_fromAblation.nonzero()
    dists_nonzero = dists_fromAblation[ix]
    fig3, ax3 = plt.subplots()
    plt.hist(dists_nonzero/255, ec='darkgrey')
    plt.title('Histogram Euclidean Distances. Tumor to Ablation. Patient ' + str(idx))
    plt.rc('figure', titlesize=25) 
    plt.xlabel('[mm]')
    plt.tick_params(labelsize=12)
    gh.save(figpathHist, width=12, height=10)
    
    
    figName_hist = pats[idx] + 'histogramDistancesfromTumortoAblation'
    figpathHist2 = os.path.join(rootdir, figName_hist)
    ix = dists_fromTumor.nonzero()
    dists_nonzero = dists_fromTumor[ix]
    fig4, ax4 = plt.subplots()
    plt.hist(dists_nonzero/255, ec='darkgrey')
    plt.title('Histogram Euclidean Distances. Ablation to Tumor. Patient ' + str(idx))
    plt.rc('figure', titlesize=25) 
    plt.xlabel('[mm]')
    plt.tick_params(labelsize=12)
    gh.save(figpathHist2, width=12, height=10)


#%%
''' save to excel '''
df_final = pd.concat([df_patientdata, df_metrics], axis=1)
filepathExcel = os.path.join(rootdir, 'SegmentationMetrics_Pooled.xlsx')
writer = pd.ExcelWriter(filepathExcel)
df_final.to_excel(writer, index=False, float_format='%.2f')

