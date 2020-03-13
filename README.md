# Radiomics Extraction and Evaluation based on Binary 3D Segmentation Images

The segmentations used in this project represent a pair of the type tumor - ablation for liver cancer tumors.  The ablation can be replaced with another type of segmentation, such as a predicted tumor segmentation (using a separate algorithm). 

The evaluation metrics include:
-  Euclidean Distances between two objects (eg. Ground Truth segmentation of Tumor vs. Predicted Segmentation of Tumor, Segmentation of Ablation vs. Segmentation of Tumor). The Mauerer et Al. algorithm for calculating euclidean distances between two objects from binary images from the SimpleITK Implementation.
- Volume Metrics (Total Volume in ml, Dice, Jaccard score, Volume Similarity, Volume Overlap Error,Residual Volume)
- Minimum Inscribed Ellipsoid and Maximum Enclosing Ellipsoid Volumes (in ml)
- PyRadiomics Metrics : https://pyradiomics.readthedocs.io/en/latest/ and https://www.radiomics.io/pyradiomics.html 


## Requirements
The following non-standard libraries are required to use the full functionality of the project.
* SimpleITK
* PyRadiomics
* PyDicom

## Main Functions and Operations Performed
1. `A_read_files_info.py` (optional)
    - `DicomReader.py`
2. `B_ResampleSegmentations.py`
3. `C_mainDistanceVolumeMetrics.py`
    - `VolumeMetrics.py`
    - `DistanceMetrics.py`
4. `D_compile_population_radiomics.py ` (optional)
    - if batch processing (for multiple segmentations) have been enabled this script will compile all the features from the outpt CSV files into one single file.
5. `E_radiomics_stats.py` (optional)
    - various plots and statistics for the features that have been previously extracted

## Usage
The scripts are called (internally) in alphabetical order using the folllowing logic:  

    Read Images --> Resample --> Extract Distance Metrics  --> Extract Volume Metrics  --> Plot Distance Metrics --> Output Metrics to Xlsx file in Tabular format

`PyRadiomics` automatically checks if the source CT image and the derived mask have the same dimensions. If not, resampling is performed in the background.
This function only operates with the path to the patient folder that can have all source CT image, segmentation masks, other files, all in one folder. It does that by creating a dictionary of paths based on the metadata information that was encoded in the [**ReferencedImageSequenceTag**](https://dicom.innolitics.com/ciods/basic-structured-display/structured-display-image-box/00720422/00081140) and [**SourceImageSequence**](https://dicom.innolitics.com/ciods/rt-beams-treatment-record/general-reference/00082112). I have previously encoded the mapping information (which is also an anonymization pipeline) in the [DICOM-Anonymization-Segmentation-Mapping](https://github.com/raluca-san/python-util-scripts/blob/master/A_fix_segmentations_dcm.py)
## Input
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--rootdir", required=False, help="path to the patient folder to be processed")
    ap.add_argument("-o", "--plots_dir", required=True, help="path to the output images")
    ap.add_argument("-b", "--input_batch_proc", required=False, help="input csv file for batch processing")
    args = vars(ap.parse_args())
The main function where to run the program from is `A_read_files_info.py`.
The function can either work with a single patient image folder by calling the function like:

`python A_read_files_info.py --i "C:\Users\MyUser\MyPatientFolderwithDicomAndSegmentationImages --o "C:\OutputFilesandImages"`  

For **Batch Processing** option the input is an Excel (.xlsx) file with the following headers:  
| Patient_ID    | Ablation_IR_Date |   Nr_Lesions | Patient_Dir_Paths                    |
| ------------- | ---------------- | ------------ | ------------------                   | 
| C001          | 20160103         |    2         | ['D:\\Users\\User1\\Pats\\Pat_C001'] |    
| C002          | 20181108         |    1         | ['D:\\Users\\User1\\Pats\\Pat_C002'] |    


The algorithm starts by iterating through all the patient folders provided in the column "Patient_Dir_Paths". Of course it's not absolutely necessarry to use the data structured in the way I did. Moreover, the `A_read_files_info.py` can be skipped altogether, especially when you know the mapping between your **Source (original) CT image -> Segmentation1 -> Segmentation2**. In this specific case my Segmentation1 is called "tumor_segmentation" and Segmentation2 is called "ablation_segmentation", which are 2 separate structures/tissue within my organ of interest (that's the liver). If you already know the file mapping between your images (i.e. which image is related to which image, aka more explicit, from which CT source image comes each segmentation) you can move the next steps which are `B_ResampleSegmentations.py` and `C_mainDistanceVolumeMetrics.py`.

## Resampling (Resizing)
In my case resampling was necessary because not only my 2 segmentation masks had different sizes [x, y, z] , but they also had different spacings in-between the slices. If your images differ only in size then you can use the `SimpleITK Paste Image` function. However, if the spacing differs, resampling is necesarry otherwise we cannot compare the 2 images using the Distance and Volume metrics.  

The file that does the resampling and resizing job is `B_ResampleSegmentations.py`. First the images need to be read and loaded using the `SimpleITK` Python library. My segmentations and CT images were saved in multiple DICOM slices (images) in a DICOM folder. The script `DicomReader.py` reads all the DICOM slices of an image/segmentation and returns a single variable which is an object in SimpleITK format. The segmentation masks are resampled in the same dimensions and spacing (however still anisotropic) before calling the `DistanceMetrics.py` and `VolumeMetrics.py`.  

This resampling script can be called like:  
   ` import SimpleITK as sitk`  
   ` import DicomReader as Reader`  
   ` tumor_segmentation_sitk, tumor_sitk_reader = Reader.read_dcm_series(tumor_path, True)`  
   ` ablation_segmentation_sitk, ablation_sitk_reader = Reader.read_dcm_series(ablation_path, True)`  
   ` resizer = ResizeSegmentation(ablation_segmentation_sitk, tumor_segmentation_sitk) # object instantiation`  
   ` tumor_segmentation_resampled = resizer.resample_segmentation()  # SimpleITK image object`  
   
     
The actual resampling operation is performed using the `SimpleITK ResampleImageFilter` and `NearestNeighbour Interpolation` such that no new segmentation mask labels are generated. 

## Segmentation Evaluation Metrics
The segmentation Evaluation Metrics are called from the script `C_mainDistanceVolumeMetrics.py` which calls:
* `DistanceMetrics.py`
* `VolumeMetrics.py`
* `RadiomicsMetrics` - features extracted using the Python library `PyRadiomics`. You can find a list of all the featuers that can be computed using this PyRadiomics [here](https://pyradiomics.readthedocs.io/en/latest/features.html). I used only the volumes, sphericity, intensity and axis values.
The same as for **Resampling**, both these scripts take as input arguments SimpleITK image objects. They can be called for example like:  

 `surface_distance_metrics = DistanceMetrics(ablation_segmentation, tumor_segmentation_resampled)`  
 `ablation_radiomics_metrics = RadiomicsMetrics(source_ct_ablation, ablation_segmentation)`  
 `evaloverlap = VolumeMetrics()`   
 
  `evaloverlap.set_image_object(ablation_segmentation, tumor_segmentation_resampled)`  
  
 
## Output
The output is a Excel (.xlsx) file in tabular format that returns radiomics (feature values) per patient and per lesion.
Aditionally a histogram that describes the Euclidean distances between the tumor and ablation (my 2 segmentations files) are generated. The script for plotting the histogram uses the Surface Euclidean Distances extracted using SimpleITK using [the Maurer et. al algorithm](https://itk.org/SimpleITKDoxygen/html/classitk_1_1simple_1_1SignedMaurerDistanceMapImageFilter.html) is in `scripts/plot_ablation_margin_hist.py`.
To compute the histogram the following steps are followed for the Maurer et. al algorithm:
1. compute the contour surface of the object (face+edge+vertex connectivity : fullyConnected=True, face connectivity only : fullyConnected=False (default mode)
2. convert from SimpleITK format to Numpy Array Img
3. remove the zeros from the contour of the object, NOT from the distance map
4. compute the number of 1's pixels in the contour
5. instantiate the Signed Mauerer Distance map for the object (negative numbers also)
6. Multiply the binary surface segmentations with the distance maps. The resulting distance maps contain non-zero values only on the surface (they can also contain zero on the surface)

![histogram_example](https://user-images.githubusercontent.com/20581812/76539679-610ca980-6481-11ea-9462-646d5620b559.png)

#### Patient Data Structure. 
The patient data consists of files and folder has the following folder structure and organization:  

`Patient_C001  

            |Series_1  
            
                     |CAS-Recordings (* .xml files)  
                     
                        | 2016-09-13_07-15-32 ...
                                                  |Segmentations
                                                                |0001.dcm
                     |CT.1.2.392...dcm
                     |CT.1.2.393...dcm
                     |CT.1.2.3..dcm  
            |Series_2
            |Series_3`  
            


