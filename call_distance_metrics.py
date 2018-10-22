import pandas as pd
import mainDistanceVolumeMetrics as Metrics
pd.options.mode.chained_assignment = None

df_final = pd.read_excel(r"C:\PatientDatasets_GroundTruth_Database\Stockholm\maverric_processed_no_registration\Filepaths_Resized_GTSegmentations_Stockholm_June.xlsx")

# df_new1 = df_final[[' Ablation Segmentation Path Resized',
#                   ' Tumour Segmentation Path Resized',
#                   'PatientID',
#                   'TrajectoryID',
#                   'Pathology']]
# df_final.rename(columns={' Ablation Segmentation Path Resized': ' Ablation Segmentation Path',
#                         ' Tumour Segmentation Path Resized': ' Tumour Segmentation Path'}, inplace=True)

rootdir = r"C:\PatientDatasets_GroundTruth_Database\Stockholm\maverric_processed_no_registration\Plots"
Metrics.main_distance_volume_metrics(df_final, rootdir)