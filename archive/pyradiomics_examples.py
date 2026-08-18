# -*- coding: utf-8 -*-
"""
@author: Raluca Sandu
"""
import SimpleITK as sitk
import six
import DicomReader
import radiomics
import B_ResampleSegmentations

# inputImage = DicomReader.read_dcm_series(r"", reader_flag=False)
# maskImage = DicomReader.read_dcm_series(r"", reader_flag=False)
inputImage = DicomReader.read_dcm_series(r"", reader_flag=False)
maskImage = DicomReader.read_dcm_series(r"", reader_flag=False)

resizer = B_ResampleSegmentations.ResizeSegmentation(inputImage, maskImage)
mask_resampled = resizer.resample_segmentation()

inputImage = sitk.Cast(inputImage, sitk.sitkFloat32)
mask_resampled = sitk.Cast(mask_resampled, sitk.sitkInt64)
mask_resampled_arr = sitk.GetArrayFromImage(mask_resampled)

tumor_surface_array_NonZero = mask_resampled_arr.nonzero()

num_tumor_surface_pixels = len(list(zip(tumor_surface_array_NonZero[0],
                                             tumor_surface_array_NonZero[1],
                                             tumor_surface_array_NonZero[2])))
# check if there is actually an object present
if 0 >= num_tumor_surface_pixels:
    raise Exception('The tumor mask image does not seem to contain an object.')

settings = {'label': 255}

extractor = radiomics.featureextractor.RadiomicsFeatureExtractor(additionalInfo=True, **settings)
result = extractor.execute(inputImage, mask_resampled)
diameter3D = result['original_shape_Maximum3DDiameter']  # euclidean
diameter2D_slice = result['original_shape_Maximum2DDiameterSlice']  # euclidean
diameter2D_col = result['original_shape_Maximum2DDiameterColumn']  # euclidean
diameter2D_row = result['original_shape_Maximum2DDiameterRow']  # euclidean
major_axis_length = result['original_shape_MajorAxisLength']  # PCA largest principal component
minimum_axis_length = result['original_shape_LeastAxisLength']

gray_lvl_nonuniformity = result['original_gldm_GrayLevelNonUniformity']
gray_lvl_variance = result['original_gldm_GrayLevelVariance']  # does it measure it in the area of the mask

print('original_shape_Maximum3DDiameter:', result['original_shape_Maximum3DDiameter'])
print('original_shape_Maximum2DDiameterSlice:', result['original_shape_Maximum2DDiameterSlice'])   # euclidean
print('original_shape_Maximum2DDiameterColumn:', result['original_shape_Maximum2DDiameterColumn'])  # euclidean
print('original_shape_Maximum2DDiameterRow:',   result['original_shape_Maximum2DDiameterRow'])  # euclidean
print('original_shape_MajorAxisLength', result['original_shape_MajorAxisLength'])  # PCA largest principal component
print('original_shape_LeastAxisLength', result['original_shape_LeastAxisLength'])

print('original_gldm_GrayLevelNonUniformity:', result['original_gldm_GrayLevelNonUniformity'])
print('original_gldm_GrayLevelVariance:', result['original_gldm_GrayLevelVariance'])  #  does it measure it in the area of the mask

# for key, val in six.iteritems(result):
#     print("\t%s: %s" %(key, val))
#%%
#
# shapeFeatures = radiomics.shape.RadiomicsShape(inputImage, mask_resampled)
# shapeFeatures.enableAllFeatures()
#
#
# print('Will calculate the following Shape features: ')
# for f in shapeFeatures.enabledFeatures.keys():
#     print('  ', f)
#     print(getattr(shapeFeatures, 'get%sFeatureValue' % f).__doc__)
#
# print('Calculating Shape features...')
# results = shapeFeatures.execute()
# print('done')
#
# print('Calculated Shape features: ')
# for (key, val) in six.iteritems(results):
#     print('  ', key, ':', val)