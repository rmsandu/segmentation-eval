# Radiomics Extraction and Evaluation based on Binary 3D Segmentation Images

The segmentations used in this project represent a pair of the type tumor - ablation.  The ablation can be replaced with another type of segmentation, such as a predicted tumor segmentation (using a separate algorithm). 

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
4. `D_compile_population_radiomics.py (optional)`
    - if batch processing (for multiple segmentations) have been enabled this script will compile all the features from the outpt CSV files into one single file.
5. `E_radiomics_stats.py` (optional)
    - various plots and statistics for the features that have been previously extracted

## Usage
The scripts are called (internally) in alphabetical order using the folllowing logic:
    * *Read Images --> Resample --> Extract Distance Metrics  --> Extract Volume Metrics  --> Plot Distance Metrics --> Output Metrics to Xlsx file in Tabular format* *
The segmentation mask are resampled in the same dimensions and spacing (isotropic) before calling the `DistanceMetrics.py` and `VolumeMetrics.py`.

`PyRadiomics` automatically checks if the source CT image and the derived mask have the same dimensions. If not, resampling is performed in the background.
This function only operates with the path to the patient folder that can have all source CT image, segmentation masks, other files, all in one folder. It does that by creating a dictionary of paths based on the metadata information that was encoded in the [**ReferencedImageSequenceTag**](https://dicom.innolitics.com/ciods/basic-structured-display/structured-display-image-box/00720422/00081140) and [**SourceImageSequence**](https://dicom.innolitics.com/ciods/rt-beams-treatment-record/general-reference/00082112). I have previously encoded the mapping information (which is also an anonymization pipeline) in the [DICOM-Anonymization-Segmentation-Mapping](https://github.com/raluca-san/python-util-scripts/blob/master/A_fix_segmentations_dcm.py)
## Input
todo
## Output
todo
### Data Preparation
The data preparation step depends on the input data to be used. (xml-processing script) (a bit dumb, move it away from that repo)
#### Single Dataset

#### Batch Processing
todo

####
    python package_data.py --intraop_patch intra_op_patch.ply  --preop_ct pre_op.ply --intraop_ct intra_op.ply --output_file packaged.pckl
#### Patient Data 
The data consists of a segmented pre-operative CT model and tracked images from a stereo-endoscope.
The data has to be organized as follows:

    .
    ├── ...
    ├── path_to_data            # input data folder
    │   ├── *.jpg               # interlaced stereo images
    │   ├── *.xml               # CASone xml parameter file
    │   └── mask                # mask folder (optional)
    │       ├── *.jpg           # segmentation masks with same name as stereo images
    └── ...

* 
#### Compute Radiomics
    python compute_stereo.py --input_dir path_to_data --result_dir path_to_output --segment true
    
* to be completed
* to be completed
* to be completed

### Plots
todo
### Stats
    todo
