# Radiomics Extraction and Evaluation based on Binary 3D SegmentationImages

- segmentations include masks of liver structures such tumors, ablations, vessels and other structures
- metrics include the Euclidean Distance between two objects (eg. Ground Truth segmentation of Tumor vs. Predicted Segmentation of Tumor, Segmentation of Ablation vs. Segmentation of Tumor)
- Mauerer et Al. algorithm for calculating euclidean distances between two objects from binary images
- Volume Metrics (DICE, Coverage,ETC)

## Requirements
The following non-standard libraries are required to use the full functionality of the project.
* SimpleITK
* PyRadiomics
* PyDicom
* Untangle
* numpy

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
