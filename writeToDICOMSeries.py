# -*- coding: utf-8 -*-
"""
Created on Thu May  3 15:40:46 2018

@author: Raluca Sandu
# Read the original series. First obtain the series file names using the
# image series reader. Then read each image using an image reader that is
# set to load all DICOM tags (public+private). The resulting images contain
# their DICOM meta-data dictionaries.
"""
import os
import time
import uuid
import hashlib
from random import random
import SimpleITK as sitk


def make_uid(entropy_srcs=None, prefix='2.25.'):
    """Generate a DICOM UID value.
    Follows the advice given at:
    http://www.dclunie.com/medical-image-faq/html/part2.html#UID
    Parameters
    ----------
    entropy_srcs : list of str or None
        List of strings providing the entropy used to generate the UID. If
        None these will be collected from a combination of HW address, time,
        process ID, and randomness.
    prefix : prefix
    """
    # Combine all the entropy sources with a hashing algorithm
    if entropy_srcs is None:
        entropy_srcs = [str(uuid.uuid1()),  # 128-bit from MAC/time/randomness
                        str(os.getpid()),  # Current process ID
                        random().hex()  # 64-bit randomness
                       ]
    hash_val = hashlib.sha256(''.join(entropy_srcs))
    # Convert this to an int with the maximum available digits
    avail_digits = 64 - len(prefix)
    int_val = int(hash_val.hexdigest(), 16) % (10 ** avail_digits)

    return prefix + str(int_val)


class DicomWriter:

    def __init__(self, img_pasted, img_source, folder_output, file_name, patient_id):
        """
        :type img_pasted: re-sized binary segmentation image in SimpleITK format
        :type img_source: source image in SimpleITK format
        :type folder_output: folder path to write the DICOM Series Files
        :type file_name: string specifying the filename, ablation or tumor z.B.
        :type patient_id: string number denoting unique patient ID
        """
        self.img_pasted = img_pasted
        self.img_source = img_source
        self.folder_output = folder_output
        self.patient_id = patient_id
        self.series_instance_uid = make_uid()
        self.study_instance_uid = make_uid()
        self.file_name = file_name

    def save_image_to_file(self):
        # Use the study/seriers/frame of reference information given in the meta-data
        # dictionary and not the automatically generated information from the file IO
        writer = sitk.ImageFileWriter()
        writer.KeepOriginalImageUIDOn()
        modification_time = time.strftime("%H%M%S")
        modification_date = time.strftime("%Y%m%d")
        direction = self.img_pasted.GetDirection()
        spacing = self.img_pasted.GetSpacing()

        series_tag_values = [("0010|0020", self.patient_id),  # set patientID
                             ("0008|0031", modification_time),  # Series Time
                             ("0008|0021", modification_date),  # Series Date
                             ("0008,0016", '1.2.840.10008.5.1.4.1.1.66.4'),  # SOP Segmentation Class UID
                             ("0020|000e", self.series_instance_uid + modification_time),  # Series Instance ID
                             ("0020|000D", self.study_instance_uid + modification_time),  # Study Instance ID
                             ("0008|0008", "DERIVED\\SECONDARY"),  # Image Type
                             # Series Instance UID
                             ("0020|000e", "1.2.826.0.1.3680043.2.1125." + modification_date + ".1" + modification_time),
                             # Image Orientation (Patient)
                             ("0020|0037", '\\'.join(map(str, (direction[0], direction[3], direction[6],
                                                               direction[1], direction[4], direction[7])))),
                             # Pixel Spacing
                             ("0028|0030", '\\'.join(map(str, (spacing[0], spacing[1], spacing[2])))),
                             ("0008|103e", "BINARY SEGMENTATION MASK")]  # Series Description

        # set patient sex ,Bits Stored, Bits Allocated, Samples per Pixel
        # set SliceThickness "0018|0050", set PatientPosition  "0018|5100"
        # set Size

        for i in range(self.img_pasted.GetDepth()):
            image_slice = self.img_pasted[:, :, i]
            # Tags shared by the series
            for tag, value in series_tag_values:
                # Slice specific tags
                image_slice.SetMetaData(tag, value)
            image_slice.SetMetaData("0008|0012", time.strftime("%Y%m%d"))  # Instance Creation Date
            image_slice.SetMetaData("0008|0013", time.strftime("%H%M%S"))  # Instance Creation Time
            # Setting the type to CT preserves the slice location.
            image_slice.SetMetaData("0008|0060", "CT")  # set the type to CT so the thickness is carried over
            # (0020, 0032) image position patient determines the 3D spacing between slices.
            # Image Position (Patient)
            image_slice.SetMetaData("0020|0032", '\\'.join(map(str, self.img_pasted.TransformIndexToPhysicalPoint((0, 0, i)))))
            image_slice.SetMetaData("0020,0013", str(i))  # Instance Number

            # Write to the output directory and add the extension dcm, to force writing in DICOM format.
            writer.SetFileName(os.path.normpath(self.folder_output + '/' + self.file_name + str(i) + '.dcm'))
            writer.Execute(image_slice)
