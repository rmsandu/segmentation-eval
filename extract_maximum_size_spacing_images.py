
import numpy as np
import DicomReader as Reader
import SimpleITK as sitk

def extract_maxSizeSpacing(ablation_paths, tumor_paths, folder_path_plan, folder_path_validation):
    # %% extract maximum size and spacing from all the images
    data = []
    for idx, ablation_path in enumerate(ablation_paths):
        if not (str(tumor_paths[idx]) == 'nan') and not (str(ablation_paths[idx]) == 'nan'):  # if both paths exists
            tumor_mask, tumor_reader = Reader.read_dcm_series(tumor_paths[idx])
            source_img_plan, img_plan_reader = Reader.read_dcm_series(folder_path_plan[idx])
            ablation_mask, ablation_reader = Reader.read_dcm_series(ablation_paths[idx])
            source_img_validation, img_validation_reader = Reader.read_dcm_series(folder_path_validation[idx])
            # check if any of the variables are empty, then skip, else resize the metrics
            # execute the condition when true and all image sources could be read
            if not (not (tumor_mask and ablation_mask and source_img_plan and source_img_validation)):
                data.append(source_img_plan)
                data.append(source_img_validation)
                data.append(ablation_mask)
                data.append(tumor_mask)


    dimension = source_img_plan.GetDimension()  #
    reference_size_x = 512
    reference_physical_size = np.zeros(dimension)
    for img in data:
        reference_physical_size[:] = [(sz - 1) * spc if sz * spc > mx else mx for sz, spc, mx in
                                      zip(img.GetSize(), img.GetSpacing(), reference_physical_size)]

    reference_spacing = [reference_physical_size[0] / (reference_size_x - 1)] * dimension

    reference_size = [int(phys_sz / (spc) + 1) for phys_sz, spc in zip(reference_physical_size, reference_spacing)]

    return reference_size, reference_spacing
