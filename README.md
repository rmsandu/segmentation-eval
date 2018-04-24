# segmentation-eval
- metrics for evaluating the quality of segmentations
- segmentations include masks of liver structures such tumors, ablations, vessels and other structures
- metrics include the Euclidean Distance between two objects (eg. Ground Truth segmentation of Tumor vs. Predicted Segmentation of Tumor, Segmentation of Ablation vs. Segmentation of Tumor)
- Mauerer et Al. algorithm for calculating euclidean distances between two objects from binary images
- Volume Metrics (DICE, Coverage,ETC)