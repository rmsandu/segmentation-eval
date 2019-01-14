# -*- coding: utf-8 -*-
"""
Created on Wed May 23 14:42:27 2018

@author: Raluca Sandu
"""

import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
import DicomReader as Reader
import DicomWriter as DicomWriter
from recordclass import recordclass
import PasteROI_Segmentation2OriginalSize as PasteROI
from extract_maximum_size_spacing_images import extract_maxSizeSpacing


class ResizeSegmentations:

    @staticmethod
    def print_image_dimensions(image, name):
        """
        Print Image Parameters.
        :param image: SimpleITK object image
        :param name: Name of the image as defined by the user
        :return:
        """
        print(name + ' direction: ', image.GetDirection())
        print(name + ' origin: ', image.GetOrigin())
        print(name + ' spacing: ', image.GetSpacing())
        print(name + ' size: ', image.GetSize())
        print(name + ' pixel: ', image.GetPixelIDTypeAsString())

    def __init__(self, df_folderpaths, root_path_to_save, flag_extract_max_size, flag_resize_only_segmentations):
        self.df_folderpaths = df_folderpaths
        self.root_path_to_save = root_path_to_save
        self.flag_extract_max_size = flag_extract_max_size
        self.flag_resize_only_segmentations =  flag_resize_only_segmentations
        self.flag_tumor = None
        self.flag_ablation = None
        self.new_filepaths = []

    def save_DICOMseries_todisk(self, images_resized, images_readers, patient_name, NeedleNr):
        """ Create directories and call DicomWriter to write the resized/resampled images to disk.

        :param images_resized: tuple of 4 SimpleITK resized images
        :param images_readers: tuple of SimpleITK Series Readers (needed to copy the metadata to the newly resized imgs)
        :param patient_name: Patient Name
        :param NeedleNr: Needle Trajectory as defined in CAS-One IR
        :return:
        """

        # create filepaths to save the new images first
        filepaths = recordclass('filepaths',
                                ['child_directory_img_plan',
                                 'child_directory_img_validation',
                                 'child_directory_ablation',
                                 'child_directory_tumor'])

        parent_directory = os.path.join(self.root_path_to_save,
                                        "Pat_GTDB_" + str(patient_name) + '_' + str(patient_name))
        child_directory_trajectory = os.path.join(parent_directory, "Trajectory" + str(NeedleNr))
        child_directory_tumor = os.path.join(parent_directory,
                                             child_directory_trajectory,
                                             "Resized_Tumor_Segmentation")
        child_directory_ablation = os.path.join(parent_directory,
                                                child_directory_trajectory,
                                                "Resized_Ablation_Segmentation")
        child_directory_img_plan = os.path.join(parent_directory, 'CT_Planning')
        child_directory_img_validation = os.path.join(parent_directory, 'CT_Validation')

        tuple_filepaths = filepaths(child_directory_img_plan=child_directory_img_plan,
                                    child_directory_img_validation=child_directory_img_validation,
                                    child_directory_ablation=child_directory_ablation,
                                    child_directory_tumor=child_directory_tumor
                                    )
        if not os.path.exists(parent_directory):
            os.makedirs(parent_directory)
            os.makedirs(child_directory_trajectory)
            os.makedirs(child_directory_tumor)
            os.makedirs(child_directory_ablation)
            os.makedirs(child_directory_img_plan)
            os.makedirs(child_directory_img_validation)
        else:
            # condition when the patient folder already exists.
            # create new trajectory folder with lesion and ablation folder, img_plan and img_validation
            if not os.path.exists(child_directory_img_validation):
                os.makedirs(child_directory_img_validation)
            if not os.path.exists(child_directory_img_plan):
                os.makedirs(child_directory_img_plan)
            if not os.path.exists(child_directory_trajectory):
                os.makedirs(child_directory_trajectory)
            if not os.path.exists(child_directory_tumor):
                os.makedirs(child_directory_tumor)
                os.makedirs(child_directory_ablation)

            # %% call DicomWriter to write to Series
        filenames = ['CT_Plan', 'CT_Validation', 'ablationSegm', 'tumorSegm']
        for i in range(len(images_resized)):
            obj_writer = DicomWriter.DicomWriter(image=images_resized[i],
                                                 folder_output=tuple_filepaths[i],
                                                 file_name=filenames[i],
                                                 patient_id=str(patient_name),
                                                 series_reader=images_readers[i])
            obj_writer.save_source_img_to_file()
        # TODO: add resized original images to the CSV list
        if self.flag_tumor is False:
            child_directory_tumor = None
        if self.flag_ablation is False:
            child_directory_ablation = None
        dict_paths = {
            "Tumour Segmentation Path Resized": child_directory_tumor,
            "Ablation Segmentation Path Resized": child_directory_ablation
        }
        self.new_filepaths.append(dict_paths)

    def I_call_resize_resample_all_images(self):
        """
            Main function that iterates through columns of the DataFrame/Excel and reads images for resizing:
            Calls DicomReader and stores the original image_plan, original image_validation, tumor_mask , ablation_mask
            in images.
        Resampled and resized images are stored in images_resized.
        :return: nothing
        """
        ablation_paths = self.df_folderpaths['AblationPath'].tolist()
        tumor_paths = self.df_folderpaths['TumorPath'].tolist()
        folder_path_plan = self.df_folderpaths['PlanTumorPath'].tolist()
        patients_names = self.df_folderpaths['PatientName'].tolist()
        folder_path_validation = self.df_folderpaths['ValidationAblationPath'].tolist()
        NeedleNr = self.df_folderpaths['NeedleNr'].tolist()
        patients_IDs = self.df_folderpaths['PatientID'].tolist()

        if self.flag_extract_max_size is True:
            # function to get the maximum spacing and size (x, y, z) from all the images
            reference_size_max, reference_spacing_max = extract_maxSizeSpacing(ablation_paths, tumor_paths,
                                                                               folder_path_plan, folder_path_plan)
        elif self.flag_resize_only_segmentations is True:
            reference_size_max = [512, 512, 400]
            # reference_size_max = [None, None, None]
            reference_spacing_max = [1.0, 1.0, 1.0]
            # get it from the segmentations assuming that tumor and its respective ablation have the same spacing settings from using the same scanner
        else:
            reference_spacing_max = [1.0, 1.0, 1.0]
            reference_size_max = [512, 512, 401]

        for idx in range(0, len(ablation_paths)):
            self.flag_tumor = False
            self.flag_ablation = False
            if not (str(tumor_paths[idx]) == 'nan') and not (str(ablation_paths[idx]) == 'nan'):  # if both paths exists
                # if patients[idx] == 195107010794: # multiple image series in the same folder
                tumor_mask, tumor_reader = Reader.read_dcm_series(tumor_paths[idx], True)
                source_img_plan, img_plan_reader = Reader.read_dcm_series(folder_path_plan[idx], True)
                ablation_mask, ablation_reader = Reader.read_dcm_series(ablation_paths[idx], True)
                source_img_validation, img_validation_reader = Reader.read_dcm_series(folder_path_validation[idx], True)
                # execute the condition when true and all image sources could be read.
                if not (not (tumor_mask and ablation_mask and source_img_plan and source_img_validation)):
                    # resize the Segmentation Mask to the dimensions of the source images they were derived from
                    self.flag_tumor = True
                    self.flag_ablation = True
                    tuple_imgs = recordclass('tuple_imgs',
                                             ['img_plan', 'img_validation',
                                              'ablation_mask', 'tumor_mask'])
                    tuple_readers = recordclass('tuple_readers', ['img_plan_reader', 'img_validation_reader',
                                                                  'ablation_reader', 'tumor_reader'])
                    images = tuple_imgs(source_img_plan, source_img_validation, ablation_mask, tumor_mask)
                    images_readers = tuple_readers(img_plan_reader, img_validation_reader, ablation_reader,
                                                   tumor_reader)

                    # resize images and segmentations to isotropic and same physical space
                    images_resized = ResizeSegmentations.II_resize_resample_images(self, images, reference_spacing_max,
                                                                                   reference_size_max, tumor_paths[idx],
                                                                                   print_flag=True)
                    # save new resized images
                    self.save_DICOMseries_todisk(images_resized, images_readers, patients_names[idx], NeedleNr[idx])
                    return # return when only patient is wanted for computation
                    # TODO: remove return if parsing of all patients is desired.



    def II_resize_resample_images(self, images, reference_spacing, reference_size, path, print_flag=False):
        """
            Resize all the images to the same dimensions and space.
            Usage: newImage = resize_image(tuple_images(img_plan, img_validation, ablation_mask, tumor_mask), [1.0, 1.0, 1.0], [512, 512, 500])
            1. Create a Blank (black) Reference IMAGE where we will "copy" all the rest of the images
            2. Paste (same spacing) or Resample (different spacing) GT Segmentations to the same size as the Original Images
            3. Resample all 4 images in the same spacing and physical space
        :param images: tuple of 4 images
        :param reference_spacing: new spacing e.g [1.0, 1.0, 1.0]. we assume isotropic spacing is to be desired
        :param reference_size: new reference size [x, y, z] for 3D images. e.g. [512, 512, 371]
        :param print_flag: print_flag default value = False. whether to print the dimensions of the imgs
        :return: resized_imgs . tuple with the 4 resized images
        """
        # %% Define tuple to store the images
        tuple_resized_imgs = recordclass('tuple_resized_imgs',
                                         ['img_plan',
                                          'img_validation',
                                          'ablation_mask',
                                          'tumor_mask'])
        # %% Create Reference image with zero origin, identity direction cosine matrix and isotropic dimension
        dimension = images.img_plan.GetDimension()  #
        reference_direction = np.identity(dimension).flatten()
        reference_origin = np.zeros(dimension)
        reference_image = sitk.Image(reference_size, images.img_plan.GetPixelID())
        reference_image.SetOrigin(reference_origin)
        reference_image.SetSpacing(reference_spacing)
        reference_image.SetDirection(reference_direction)
        reference_center = np.array(
            reference_image.TransformContinuousIndexToPhysicalPoint(np.array(reference_image.GetSize()) / 2.0))

        # %% PRINT DIMENSIONS
        if print_flag:
            ResizeSegmentations.print_image_dimensions(reference_image, 'REFERENCE IMAGE')
            ResizeSegmentations.print_image_dimensions(images.tumor_mask, 'TUMOR Original')
            ResizeSegmentations.print_image_dimensions(images.ablation_mask, 'ABLATION Original')
            ResizeSegmentations.print_image_dimensions(images.img_plan, 'IMAGE PLAN ORIGINAL')
            ResizeSegmentations.print_image_dimensions(images.img_validation, 'IMAGE VALIDATION ORIGINAL')

        # %% RESIZE the ROI SEGMENTATIONS before RESAMPLING and RESIZING TO REFERENCE IMAGE
        if  self.flag_resize_only_segmentations is True:
            # use the maximum size
            images.tumor_mask = PasteROI.paste_roi_image(images.img_plan, images.tumor_mask, reference_size)
            images.ablation_mask = PasteROI.paste_roi_image(images.img_plan, images.ablation_mask, reference_size)
        else:
            if ( images.img_plan.GetSpacing() - images.tumor_mask.GetSpacing() ) < 0.05:
                # use PASTE if the original img and segmentation have the similar spacing
                # TODO: maybe use reference size here - standard no of slices for all segmentations
                images.tumor_mask = PasteROI.paste_roi_image(images.img_plan, images.tumor_mask)
                print('-----Plan and mask match----', path)
            else:
                # use RESAMPLE if the original image and its segmentation have different spacing
                print('different spacing of tumor segmentation and original image: ', path)
                ResizeSegmentations.print_image_dimensions(images.tumor_mask, 'TUMOR ORIGINAL')
                ResizeSegmentations.print_image_dimensions(images.img_plan, 'IMAGE PLAN ORIGINAL')
                images.tumor_mask = PasteROI.resize_segmentation(images.img_plan, images.tumor_mask)

            if images.img_validation.GetSpacing() == images.ablation_mask.GetSpacing():
                images.ablation_mask = PasteROI.paste_roi_image(images.img_validation, images.ablation_mask)
                print('-----Validation and mask match----', path)
            else:
                # use RESAMPLE if the original image and its segmentation have different spacing
                print('different spacing of ablation segmentation and original image: ', path)
                ResizeSegmentations.print_image_dimensions(images.ablation_mask, 'ABLATION ORIGINAL')
                ResizeSegmentations.print_image_dimensions(images.img_validation, 'IMAGE VALIDATION ORIGINAL')
                images.ablation_mask = PasteROI.resize_segmentation(images.img_validation, images.ablation_mask)

        # PRINT DIMENSIONS AFTER RESIZING GT Segmentation MASKS ROI
        if print_flag:
            ResizeSegmentations.print_image_dimensions(images.tumor_mask, 'TUMOR ROI RESIZED')
            ResizeSegmentations.print_image_dimensions(images.ablation_mask, 'ABLATION ROI RESIZED')

        # %%  APPLY TRANSFORMS TO RESIZE AND RESAMPLE ALL THE DATA IN THE SAME SPACE
        if self.flag_resize_only_segmentations is True:
            # only resize the segmentations intra-patient for computation of metrics
            resized_imgs = tuple_resized_imgs(img_plan=images.img_plan,
                                              img_validation=images.img_validation,
                                              ablation_mask=images.ablation_mask ,
                                              tumor_mask=images.tumor_mask )
        else:
            # transform all the images in the same space, size & spacing
            data = [images.img_plan, images.img_validation, images.ablation_mask, images.tumor_mask]
            data_resized = []
            for idx, img in enumerate(data):
                # Set Transformation
                transformTranslation = sitk.AffineTransform(dimension)  # use affine transform with 3 dimensions
                transformTranslation.SetMatrix(img.GetDirection())  # set the cosine direction matrix
                transformTranslation.SetTranslation(np.array(img.GetOrigin() - reference_origin))
                transformTranslation.SetCenter(reference_center)
                centering_transform = sitk.TranslationTransform(dimension)
                img_center = np.array(img.TransformContinuousIndexToPhysicalPoint(np.array(img.GetSize()) / 2.0))
                centering_transform.SetOffset(
                    np.array(transformTranslation.GetInverse().TransformPoint(img_center) - reference_center))
                centered_transform = sitk.Transform(transformTranslation)
                centered_transform.AddTransform(centering_transform)
                # set all  output image parameters: origin, spacing, direction, starting index, and size with RESAMPLE
                resampler = sitk.ResampleImageFilter()
                resampler.SetReferenceImage(reference_image)
                resampler.SetTransform(centered_transform)
                resampler.SetDefaultPixelValue(0)
                if idx == 0 or idx == 1:
                    resampler.SetInterpolator(sitk.sitkLinear)
                elif idx == 2 or idx == 3:
                    # use NearestNeighbor interpolation for the ablation&tumor segmentations so no new labels are generated
                    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
                resampled_img = resampler.Execute(img)
                data_resized.append(resampled_img)
                if print_flag:
                    ResizeSegmentations.print_image_dimensions(resampled_img, 'RESAMPLED IMAGE ' + str(idx))

            # assuming the order stays the same, reassigng back to tuple
            resized_imgs = tuple_resized_imgs(img_plan=data_resized[0],
                                              img_validation=data_resized[1],
                                              ablation_mask=data_resized[2],
                                              tumor_mask=data_resized[3])

        return resized_imgs


    def get_new_filepaths(self):
        """
        Return Updated DataFrame with new resized filepaths.
        :return: DataFrame object
        """
        df_new_filepaths = pd.DataFrame(self.new_filepaths)
        df_final = pd.concat([self.df_folderpaths, df_new_filepaths], axis=1)
        return df_final
