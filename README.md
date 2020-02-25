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


## Usage

### Data Preparation
The data preparation step depends on the input data to be used.
#### Single Dataset
tpdo
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

### Plot
For experiments with synthetic data run (add optional parameters)
### Output to CSV
    python gp_experiments.py
