# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 13:43:49 2017

Mauerer Distance Map for the tumor Object (deemed as "tumor in this particular case")
Algorithm Pipeline :
    1. compute the contour surface of the object (
    face+edge+vertex connectivity : fullyConnected=True
    face connectivity only : fullyConnected=False (default mode)
    2. convert from SimpleITK format to Numpy Array Img
    3. remove the zeros from the contour of the object, NOT from the distance map
    4. compute the number of 1's pixels in the contour
    5. instantiate the Signed Mauerer Distance map for the object (negative numbers also)
        # Multiply the binary surface segmentations with the distance maps. The resulting distance
        # maps contain non-zero values only on the surface (they can also contain zero on the surface)
@author: Raluca Sandu
"""
from enum import Enum

import SimpleITK as sitk
import numpy as np
import pandas as pd
import radiomics
from scipy import ndimage


class RadiomicsMetrics(object):

    def __init__(self, input_image, mask_image):
        self.input_image = input_image
        self.mask_image = mask_image
        self.error_flag = False

        class AxisMetricsRadiomics(Enum):
            center_of_mass_x, center_of_mass_y, center_of_mass_z, \
            center_of_mass_index_x, center_of_mass_index_y, center_of_mass_index_z, \
            elongation, sphericity, mesh_volume, intensity_mean, intensity_variance, intensity_uniformity, \
            diameter3D, diameter2D_slice, diameter2D_col, diameter2D_row, major_axis_length, \
            least_axis_length, minor_axis_length, gray_lvl_nonuniformity, gray_lvl_variance = range(21)

        axis_metrics_results = np.zeros((1, len(AxisMetricsRadiomics.__members__.items())))
        # %% Extract the diameter axis
        settings = {'label': 255, 'correctMask': True}
        extractor = radiomics.featureextractor.RadiomicsFeatureExtractor(additionalInfo=True, **settings)
        try:
            result = extractor.execute(self.input_image, self.mask_image)
        except Exception as e:
            print(repr(e))
            self.error_flag = True
            return
        try:
            axis_metrics_results[0, AxisMetricsRadiomics.center_of_mass_x.value] = \
                result['diagnostics_Mask-original_CenterOfMass'][0]
            axis_metrics_results[0, AxisMetricsRadiomics.center_of_mass_y.value] = \
                result['diagnostics_Mask-original_CenterOfMass'][1]
            axis_metrics_results[0, AxisMetricsRadiomics.center_of_mass_z.value] = \
                result['diagnostics_Mask-original_CenterOfMass'][2]
        except Exception:
            axis_metrics_results[0, AxisMetricsRadiomics.center_of_mass_x.value] = None
            axis_metrics_results[0, AxisMetricsRadiomics.center_of_mass_y.value] = None
            axis_metrics_results[0, AxisMetricsRadiomics.center_of_mass_z.value] = None
        try:
            axis_metrics_results[0, AxisMetricsRadiomics.center_of_mass_index_x.value] = \
                result['diagnostics_Mask-original_CenterOfMassIndex'][0]
            axis_metrics_results[0, AxisMetricsRadiomics.center_of_mass_index_y.value] = \
                result['diagnostics_Mask-original_CenterOfMassIndex'][1]
            axis_metrics_results[0, AxisMetricsRadiomics.center_of_mass_index_z.value] = \
                result['diagnostics_Mask-original_CenterOfMassIndex'][2]
        except Exception:
            axis_metrics_results[0, AxisMetricsRadiomics.center_of_mass_index_x.value] = None
            axis_metrics_results[0, AxisMetricsRadiomics.center_of_mass_index_y.value] = None
            axis_metrics_results[0, AxisMetricsRadiomics.center_of_mass_index_z.value] = None
        try:
            axis_metrics_results[0, AxisMetricsRadiomics.mesh_volume.value] = (result['original_shape_MeshVolume'].tolist())/1000
        except Exception:
            axis_metrics_results[0, AxisMetricsRadiomics.mesh_volume.value] = None
        # getMeshVolumeFeatureValue()
        try:
            axis_metrics_results[0, AxisMetricsRadiomics.elongation.value] = result['original_shape_Elongation']
        except Exception:
            axis_metrics_results[0, AxisMetricsRadiomics.elongation.value] = None
        try:
            axis_metrics_results[0, AxisMetricsRadiomics.sphericity.value] = result['original_shape_Sphericity']
        except Exception:
            axis_metrics_results[0, AxisMetricsRadiomics.sphericity.value] = None
        try:
            axis_metrics_results[0, AxisMetricsRadiomics.intensity_mean.value] = result['original_firstorder_Mean']
        except Exception:
            axis_metrics_results[0, AxisMetricsRadiomics.intensity_mean.value] = None
        try:
            axis_metrics_results[0, AxisMetricsRadiomics.intensity_variance.value] = result[
                'original_firstorder_Variance']
        except Exception:
            axis_metrics_results[0, AxisMetricsRadiomics.intensity_variance.value] = None
        try:
            axis_metrics_results[0, AxisMetricsRadiomics.intensity_uniformity.value] = result[
                'original_firstorder_Uniformity']
        except Exception:
            axis_metrics_results[0, AxisMetricsRadiomics.intensity_uniformity.value] = None
        try:
            axis_metrics_results[0, AxisMetricsRadiomics.diameter3D.value] = result['original_shape_Maximum3DDiameter']
        except Exception:
            axis_metrics_results[0, AxisMetricsRadiomics.diameter3D.value] = None
        try:
            axis_metrics_results[0, AxisMetricsRadiomics.diameter2D_slice.value] = result[
                'original_shape_Maximum2DDiameterSlice']  # euclidean
        except Exception:
            axis_metrics_results[0, AxisMetricsRadiomics.diameter2D_slice.value] = None
        try:
            axis_metrics_results[0, AxisMetricsRadiomics.diameter2D_col.value] = result[
                'original_shape_Maximum2DDiameterColumn']  # euclidean
        except Exception:
            axis_metrics_results[0, AxisMetricsRadiomics.diameter2D_col.value] = None
        try:
            axis_metrics_results[0, AxisMetricsRadiomics.diameter2D_row.value] = result[
                'original_shape_Maximum2DDiameterRow']  # euclidean
        except Exception:
            axis_metrics_results[0, AxisMetricsRadiomics.diameter2D_row.value] = None
        try:
            # PCA largest principal component
            axis_metrics_results[0, AxisMetricsRadiomics.major_axis_length.value] = result[
                'original_shape_MajorAxisLength']
        except Exception:
            axis_metrics_results[0, AxisMetricsRadiomics.major_axis_length.value] = None
        try:
            axis_metrics_results[0, AxisMetricsRadiomics.least_axis_length.value] = result[
                'original_shape_LeastAxisLength']
        except Exception:
            axis_metrics_results[0, AxisMetricsRadiomics.least_axis_length.value] = None
        try:
            axis_metrics_results[0, AxisMetricsRadiomics.minor_axis_length.value] = result[
                'original_shape_MinorAxisLength']
        except Exception:
            axis_metrics_results[0, AxisMetricsRadiomics.minor_axis_length.value] = None
        try:
            axis_metrics_results[0, AxisMetricsRadiomics.gray_lvl_nonuniformity.value] = result[
                'original_gldm_GrayLevelNonUniformity']
        except Exception:
            axis_metrics_results[0, AxisMetricsRadiomics.gray_lvl_nonuniformity.value] = None
        try:
            axis_metrics_results[0, AxisMetricsRadiomics.gray_lvl_variance.value] = result[
                'original_gldm_GrayLevelVariance']
        except Exception:
            axis_metrics_results[0, AxisMetricsRadiomics.gray_lvl_variance.value] = None

        # %% Save to DataFrame
        self.axis_metrics_results_df = pd.DataFrame(data=axis_metrics_results, index=list(range(1)),
                                                    columns=[name for name, _ in
                                                             AxisMetricsRadiomics.__members__.items()])

    def get_axis_metrics_df(self):
        return self.axis_metrics_results_df


class DistanceMetrics(object):

    def __init__(self, ablation_segmentation, tumor_segmentation):

        self.tumor_segmentation = tumor_segmentation
        self.ablation_segmentation = ablation_segmentation
        self.error_flag = False
        ''' init the enum fields for surface dist measures computer with simpleitk'''

        class SurfaceDistanceMeasuresITK(Enum):
            hausdorff_distance, max_distance, min_surface_distance, mean_surface_distance, \
            median_surface_distance, std_surface_distance = range(6)

        surface_distance_results = np.zeros((1, len(SurfaceDistanceMeasuresITK.__members__.items())))
        # %%
# <<<<<<< HEAD
#
#         tumor_surface = sitk.LabelContour(tumor_segmentation, fullyConnected=False)
# =======
        # tumor_surface = sitk.LabelContour(tumor_segmentation, fullyConnected=True)
        # tumor_surface = sitk.LabelContour(tumor_segmentation, fullyConnected=False)
        # tumor_surface_array = sitk.GetArrayFromImage(tumor_surface)
        # >>>>>>> fixed border extraction Iwan
        tumor_array = sitk.GetArrayFromImage(tumor_segmentation)
        border_inside = ndimage.binary_erosion(tumor_array, structure=ndimage.generate_binary_structure(3, 1))
        tumor_surface_array = tumor_array ^ border_inside


        tumor_surface_array_NonZero = tumor_surface_array.nonzero()

        self.num_tumor_surface_pixels = len(list(zip(tumor_surface_array_NonZero[0],
                                                     tumor_surface_array_NonZero[1],
                                                     tumor_surface_array_NonZero[2])))
        # check if there is actually an object present

        if 0 >= self.num_tumor_surface_pixels:
            print('The tumor mask image does not seem to contain an object.')
            self.error_flag = True
            return
        # init signed mauerer distance as tumor metrics from SimpleITK
        self.tumor_distance_map = sitk.SignedMaurerDistanceMap(tumor_segmentation,
                                                               squaredDistance=False,
                                                               useImageSpacing=True)

        hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()
        try:
            hausdorff_distance_filter.Execute(tumor_segmentation, ablation_segmentation)
        except Exception as e:
            print(repr(e))
        surface_distance_results[0, SurfaceDistanceMeasuresITK.hausdorff_distance.value] = \
            hausdorff_distance_filter.GetHausdorffDistance()

        # %%
        ''' Mauerer Distance Map for the Ablation Object '''
        ablation_surface = sitk.LabelContour(ablation_segmentation, fullyConnected=False)
        ablation_surface_mask_array = sitk.GetArrayFromImage(ablation_surface)
        ablation_mask_array_NonZero = ablation_surface_mask_array.nonzero()

        # Get the number of pixels in the mask surface by counting all pixels that are non-zero
        self.num_ablation_surface_pixels = len(list(zip(ablation_mask_array_NonZero[0],
                                                        ablation_mask_array_NonZero[1],
                                                        ablation_mask_array_NonZero[2])))

        if 0 >= self.num_ablation_surface_pixels:
            print('The ablation mask image does not seem to contain an object.')
            self.error_flag = True
            return
        # init Mauerer Distance
        self.ablation_distance_map = sitk.SignedMaurerDistanceMap(ablation_segmentation,
                                                                  squaredDistance=False,
                                                                  useImageSpacing=True)

        ablation_distance_map_array = sitk.GetArrayFromImage(self.ablation_distance_map)

        # compute the contours multiplied with the euclidean distances 
        self.tumor2ablation_distance_map = ablation_distance_map_array * tumor_surface_array

        # remove the zeros from the surface contour(indexes) from the distance maps '''
        self.surface_distances = list(self.tumor2ablation_distance_map[tumor_surface_array_NonZero] / -255)

        # %%
        ''' Compute the surface distances max, min, mean, median, std '''
        surface_distance_results[0, SurfaceDistanceMeasuresITK.max_distance.value] = np.max(self.surface_distances)
        surface_distance_results[0, SurfaceDistanceMeasuresITK.min_surface_distance.value] = \
            np.min(self.surface_distances)
        surface_distance_results[0, SurfaceDistanceMeasuresITK.mean_surface_distance.value] = \
            np.mean(self.surface_distances)
        surface_distance_results[0, SurfaceDistanceMeasuresITK.median_surface_distance.value] = \
            np.median(self.surface_distances)
        surface_distance_results[0, SurfaceDistanceMeasuresITK.std_surface_distance.value] = \
            np.std(self.surface_distances)

        # Save to DataFrame
        self.surface_distance_results_df = pd.DataFrame(data=surface_distance_results, index=list(range(1)),
                                                        columns=[name for name, _ in
                                                                 SurfaceDistanceMeasuresITK.__members__.items()])
        # change the name of the columns
        self.surface_distance_results_df.columns = ['Hausdorff_AT', 'Maximum_AT', 'Minimum_AT', 'Mean_AT', 'Median_AT',
                                                    'Std_AT']

    # %%  methods to return the distances
    def get_SitkDistances(self):
        return self.surface_distance_results_df

    def get_ablation_dist_map(self):
        return self.tumor2ablation_distance_map

    def get_surface_distances(self):
        return self.surface_distances
