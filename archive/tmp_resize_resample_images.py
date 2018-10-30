import SimpleITK as sitk
import numpy as np
import collections

def resize_resample_images(images):
    """ Resize all the images to the same dimensions, spacing and origin.
        Usage: newImage = resize_image(source_img_plan, source_img_validation, ROI(ablation/tumor)_mask)
        1. translate to same origin
        2. largest number of slices and interpolate the others.
        3. same resolution 1x1x1 mm3 - resample
        4. (physical space)
        Slice Thickness (0018,0050)
        ImagePositionPatient (0020,0032)
        ImageOrientationPatient (0020,0037)
        PixelSpacing (0028,0030)
        Frame Of Reference UID (0020,0052)
    """
    # %% Define tuple to store the images
    tuple_resized_imgs = collections.namedtuple('tuple_resized_imgs',
                                                ['img_plan',
                                                 'img_validation',
                                                 'ablation_mask',
                                                 'tumor_mask'])
    # %% Create Reference image with zero origin, identity direction cosine matrix and isotropic dimension
    dimension = images.img_plan.GetDimension()  #
    reference_spacing = np.ones(dimension) # resize to isotropic size
    reference_direction = np.identity(dimension).flatten()
    reference_size = [512] * dimension
    reference_origin = np.zeros(dimension)
    data = [images.img_plan, images.img_validation, images.ablation_mask, images.tumor_mask]
    #
    # reference_physical_size = np.zeros(dimension)
    # for img in data:
    #     reference_physical_size[:] = [(sz - 1) * spc if sz * spc > mx else mx for sz, spc, mx in
    #                                   zip(img.GetSize(), img.GetSpacing(), reference_physical_size)]
    # reference_spacing = [phys_sz / (sz - 1) for sz, phys_sz in zip(reference_size, reference_physical_size)]

    reference_image = sitk.Image(reference_size, images.img_plan.GetPixelIDValue())
    reference_image.SetOrigin(reference_origin)
    reference_image.SetSpacing(reference_spacing)
    reference_image.SetDirection(reference_direction)
    reference_center = np.array(
        reference_image.TransformContinuousIndexToPhysicalPoint(np.array(reference_image.GetSize()) / 2.0))

    print('TUMOR Original:')
    print(' image mask direction', images.tumor_mask.GetDirection())
    print('image mask origin', images.tumor_mask.GetOrigin())
    print('mask spacing', images.tumor_mask.GetSpacing())
    print('IMAGE PLAN ORIGINAL:')
    print(' mask direction', images.img_plan.GetDirection())
    print(' mask origin', images.img_plan.GetOrigin())
    print('spacing', images.img_plan.GetSpacing())
    #%% Paste the GT segmentation masks before transformation
    tumor_mask_paste = (paste_roi_image(images.img_plan, images.tumor_mask))
    ablation_mask_paste = (paste_roi_image(images.img_validation, images.ablation_mask))
    images.tumor_mask = tumor_mask_paste
    images.ablation_mask = ablation_mask_paste
    print('TUMOR PASTED:')
    print('pasted image mask direction', tumor_mask_paste.GetDirection())
    print('pasted image mask origin', tumor_mask_paste.GetOrigin())
    print('pasted image mask spacing', tumor_mask_paste.GetSpacing())
    print('ABLATION PASTED:')
    print('pasted image mask direction', ablation_mask_paste.GetDirection())
    print('pasted image mask origin', ablation_mask_paste.GetOrigin())
    print('pasted image mask spacing', ablation_mask_paste.GetSpacing())

    # %%  Apply transforms
    data_resized = []

    for idx,img in enumerate(data):
        #%% Set Transformation
        transform = sitk.AffineTransform(dimension) # use affine transform with 3 dimensions
        transform.SetMatrix(img.GetDirection()) # set the cosine direction matrix
        transform.SetTranslation(np.array(img.GetOrigin()) - reference_origin) # set the translation.
        # Modify the transformation to align the centers of the original and reference image instead of their origins.
        centering_transform = sitk.TranslationTransform(dimension)
        img_center = np.array(img.TransformContinuousIndexToPhysicalPoint(np.array(img.GetSize()) / 2.0))
        centering_transform.SetOffset(np.array(transform.GetInverse().TransformPoint(img_center) - reference_center))
        centered_transform = sitk.Transform(transform)
        centered_transform.AddTransform(centering_transform)

        #%% set all  output image parameters: origin, spacing, direction, starting index, and size.
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(reference_image)
        resampler.SetDefaultPixelValue(img.GetPixelIDValue())
        # resampler.SetTransform(centered_transform)
        resampler.SetSize(reference_image.GetSize())
        resampler.SetOutputSpacing(reference_image.GetSpacing())
        resampler.SetOutputOrigin(reference_image.GetOrigin())
        resampler.SetOutputDirection(reference_image.GetDirection())
        if idx==0 or idx==1:
            resampler.SetInterpolator(sitk.sitkLinear)
        elif idx==1 or idx==3:
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampled_img = resampler.Execute(img)


        # if (idx==2 or idx==3):
        #     # temporary solution to resample the GT image with NearestNeighbour
        #     resampled_img = sitk.Resample(img, reference_image, centered_transform, sitk.sitkNearestNeighbor, 0.0)
        # else:
        #     # CT source image
        #     resampled_img = sitk.Resample(img, reference_image, centered_transform, sitk.sitkLinear, 0.0)

        # append to list
        data_resized.append(resampled_img)

    print('TUMOR RESAMPLED:')
    print('resampled image mask direction', data_resized[0].GetDirection())
    print('resampled image mask origin', data_resized[0].GetOrigin())
    print('resampled image mask spacing', data_resized[0].GetSpacing())

    # assuming the order stays the same, reassigng back to tuple
    resized_imgs = tuple_resized_imgs(img_plan=data_resized[0],
                                      img_validation=data_resized[1],
                                      ablation_mask=data_resized[2],
                                      tumor_mask=data_resized[3])
    return resized_imgs
