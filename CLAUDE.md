# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A radiomics extraction and evaluation pipeline for pairs of binary 3D DICOM segmentation masks (typically liver tumor vs. ablation zone). It computes Euclidean surface distances, volume-overlap metrics (Dice, Jaccard, Volume Similarity, etc.), inner/outer ellipsoid approximations, and PyRadiomics shape/intensity features, then writes results to Excel.

There is no packaging (`setup.py` is empty), no test suite, no linter config, and no CI. This is a research codebase run as standalone scripts, historically from Windows paths and a SLURM cluster (`job_calc_metrics.sh`).

## Environment / running

```bash
pip install -r requirements.txt
```

Pinned to old versions (SimpleITK 1.2.4, numpy 1.22, pandas 1.0.5, PyRadiomics 3.0) — install into a dedicated virtualenv/conda env, not system Python.

There are no automated tests. `tests/` contains standalone exploratory/plotting scripts (e.g. `test_missing_lesion.py`), not pytest tests — run individually with `python tests/<script>.py` if needed, after editing the hardcoded file paths at the top.

## Pipeline entry point and flow

Scripts are named `A_`–`E_` and are meant to be run in that order; `A` is the actual program entry point and calls into `B` and `C` internally:

```
A_read_files_info.py --i <patient_dicom_root> --o <output_dir> [--b <batch_xlsx>]
```

Flow: **Read Images → Resample → Extract Distance Metrics → Extract Volume Metrics → Plot Distance Histogram → Write Excel**

1. **`A_read_files_info.py`** — entry point. Walks a patient DICOM folder tree, uses `DicomReader.py` to identify and pair CT source images with their tumor/ablation segmentation series (matched via DICOM `ReferencedImageSequence`/`SourceImageSequence` tags). Supports single-patient mode (`--i`) or batch mode over multiple patients (`--b`, an xlsx with `Patient_ID`, `Ablation_IR_Date`, `Nr_Lesions`, `Patient_Dir_Paths` columns).
2. **`B_ResampleSegmentations.py`** (`ResizeSegmentation` class) — resamples tumor/ablation masks onto the same size/spacing/origin (nearest-neighbor interpolation, so no new labels are introduced) since the two segmentations often come from differently-spaced series.
3. **`C_mainDistanceVolumeMetrics.py`** (`main_distance_volume_metrics`) — orchestrates per-lesion metric extraction, called once per tumor/ablation pair:
   - `DistanceMetrics.py`: `DistanceMetrics` class computes surface-to-surface Euclidean distances (Maurer distance transform, via `surface_distance/metrics.py` and SimpleITK); `RadiomicsMetrics` class wraps PyRadiomics shape/intensity feature extraction.
   - `VolumeMetrics.py`: `VolumeMetrics` class computes Dice/Jaccard/volume similarity/overlap error via `sitk.LabelOverlapMeasuresImageFilter`, plus tumor coverage/residual volume and inner/outer ellipsoid volumes (via `scripts/ellipsoid_inner_outer.py`, convex optimization with CVXPY).
   - `scripts/plot_ablation_margin_hist.py`: plots the color-coded histogram of surface distances.
   - Writes one Excel file per patient/lesion with all metrics combined.
4. **`D_compile_population_radiomics.py`** (optional) — merges the per-patient/lesion Excel outputs from batch runs into one population-level file.
5. **`E_radiomics_stats.py`** (optional) — downstream statistics/plots over the compiled population radiomics.

## Key modules

- **`DicomReader.py`** — reads a DICOM series folder into a `SimpleITK` image (`read_dcm_series`), or raw slices via `pydicom` (`read_dcm_series_pydicom`).
- **`DicomWriter.py`** — writes SimpleITK images back out as DICOM series.
- **`surface_distance/`** — vendored surface-distance metric library (`metrics.py`, `lookup_tables.py`) used for the Maurer-algorithm surface distance computation.
- **`customradiomics/margin.py`** — bounding-box/crop/masked-distance helpers used in margin computations, independent of the DICOM pipeline (used by `calc_margin.py`).
- **`utils/`** — grab-bag of standalone helper scripts (resampling, ROI pasting, plotting, animation, keyboard input) invoked ad hoc, not imported as a package from the main pipeline.
- **`scripts/`** — plotting and ellipsoid-fitting helpers called from the main pipeline (`ellipsoid_inner_outer.py`, `plot_ablation_margin_hist.py`) plus standalone analysis/plotting scripts run independently.
- **`import_from_csv/`** — one-off scripts for importing external clinical data (REDCap exports, chemotherapy, LTP) and joining with computed radiomics; not part of the core pipeline.
- **`archive/`** — superseded versions of scripts, kept for reference only.

## Working conventions

- Patient/lesion identity flows through the pipeline as `patient_id` + `lesion_id` (+ `ablation_date`), and most metric classes/functions take these plus SimpleITK image objects, returning a `pandas.DataFrame` row that gets concatenated into the final per-lesion Excel output — follow this pattern when adding new metrics.
- Segmentation resampling always uses nearest-neighbor interpolation to avoid introducing new label values; don't switch to linear/cubic interpolation for masks.
- Many scripts (especially in `tests/`, `scripts/`, `utils/`) hardcode Windows-style absolute file paths at the top for local runs — expect to edit these before running a given script directly, rather than looking for CLI args.
