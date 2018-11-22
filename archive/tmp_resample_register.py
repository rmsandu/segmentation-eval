import itk
import os
import numpy as np
import pandas as pd
import DicomReader as Reader
import DicomWriter as DicomWriter
import Resize_Resample_archive as PasteRoi
import SimpleITK as sitk
#%%
def resample(image, transform):
    reference_image = image
    interpolator = sitk.sitkCosineWindowedSinc
    default_value = 100.0
    return sitk.Resample(image, reference_image, transform,
                         interpolator, default_value)


def save_transform_and_image(transform, fixed_image, moving_image, outputfile_prefix):
    """
    Write the given transformation to file, resample the moving_image onto the fixed_images grid and save the
    result to file.

    Args:
        transform (SimpleITK Transform): transform that maps points from the fixed image coordinate system to the moving.
        fixed_image (SimpleITK Image): resample onto the spatial grid defined by this image.
        moving_image (SimpleITK Image): resample this image.
        outputfile_prefix (string): transform is written to outputfile_prefix.tfm and resampled image is written to
                                    outputfile_prefix.mha.
    """
    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(fixed_image)
    # SimpleITK supports several interpolation options, we go with the simplest that gives reasonable results.
    resample.SetInterpolator(sitk.sitkLinear)
    resample.SetTransform(transform)
    sitk.WriteImage(resample.Execute(moving_image), outputfile_prefix + '.mha')
    # sitk.WriteTransform(transform, outputfile_prefix + '.tfm')
    # TODO: modify the type of writing to disk


#%%
folder_path_plan = r"C:\PatientDatasets_GroundTruth_Database\Stockholm\3d_segmentation_maverric\maverric\Pat_ALKATIB SALEM_195412161795\Study_840\Series_9"
folder_path_validation = r"C:\PatientDatasets_GroundTruth_Database\Stockholm\3d_segmentation_maverric\maverric\Pat_ALKATIB SALEM_195412161795\Study_840\Series_17"

source_img_plan = Reader.read_dcm_series(folder_path_plan, True)
source_img_validation = Reader.read_dcm_series(folder_path_validation, True)
fixed_image = source_img_plan
moving_image = source_img_validation
# save_transform_and_image(transform, source_img_plan, source_img_validation, 'new_registered_file')

#%%
# dim = 3
# matrix = [m*s for m,s in zip(image_direction, image_spacing*dim)]
#
# affine = sitk.AffineTransform(matrix, (0,0,0),image_origin)
# affine.SetMatrix(matrix)
# affine.SetTranslation(image_origin)
# resampled = resample(source_img_plan, affine)
#
#%%
dimension = 3
matrix = np.array((0.999978174, 0.006592837434, -0.0004312256374, -0.00660670892, 0.9983414207, -0.05719055176,
                  5.3462405e-005, 0.0571921525, 0.9983631878, -5.15731875, -55.32705312, -2.795551313))
# define Transform
transform = sitk.Transform()
#init transform
transform.SetParameters(matrix)
# init registration with transform matrix
R = sitk.ImageRegistrationMethod()
R.SetInitialTransform(transform)
# Define interpolation method
R.SetInterpolator(sitk.sitkLinear)
# Perform registration
R.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=1)
out_img = R.Execute(sitk.Cast(fixed_image, sitk.sitkFloat32),
                                              sitk.Cast(moving_image, sitk.sitkFloat32))
#%%
resampler = sitk.ResampleImageFilter()
resampler.SetReferenceImage(fixed_image)
resampler.SetInterpolator(sitk.sitkLinear)
#resampler.SetDefaultPixelValue(100)
resampler.SetTransform(out_img)
out = resampler.Execute(moving_image)
writer_dcm = DicomWriter(folder_output=r'C:\develop\test', )
writer_dcm
# # initial_transform  = sitk.CenteredTransformInitializer(sitk.Cast(fixed_image,moving_image.GetPixelID()),
#                                                       moving_image,
#                                                       sitk.Euler3DTransform(),
#                                                       sitk.CenteredTransformInitializerFilter.GEOMETRY)
#%%
# registration_method = sitk.ImageRegistrationMethod()
#
# # registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
# # registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
# # registration_method.SetMetricSamplingPercentage(0.01)
#
# registration_method.SetInterpolator(sitk.sitkLinear)
#
# # registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100)
# # # Scale the step size differently for each parameter, this is critical!!!
# # registration_method.SetOptimizerScalesFromPhysicalShift()
#
# registration_method.SetInitialTransform(transform, inPlace=False)
#
# final_transform_v1 = registration_method.Execute(sitk.Cast(fixed_image, sitk.sitkFloat32),
#                                               sitk.Cast(moving_image, sitk.sitkFloat32))
# #
# # #%% Registration Steps
# '''
# 1. set affine matrix (its the registration matrix)
# 2. set affine transform
# 3. set center
# 4. set interpolator
