# Archive

Historical/retired code kept for reference. Nothing here is imported by the
active pipeline (`A_read_files_info.py` -> `E_radiomics_stats.py`), it is not
linted or tested, and none of it is guaranteed to run as-is. Hardcoded local
paths in these files have been replaced with generic `/path/to/...`
placeholders, but the code itself has not been modernized.

| File | Status |
|---|---|
| `SegmentationEval.py` | Monolithic pre-refactor version of the whole pipeline. Superseded by `A_read_files_info.py` -> `C_mainDistanceVolumeMetrics.py`. Imports `medpy` (not in requirements.txt) and the local `surface.py` below. |
| `surface.py` | Vendored `medpy.metric.surface.Surface` class. Superseded in the current pipeline by `surface_distance/metrics.py`. Only used by `SegmentationEval.py`. |
| `pyradiomics_examples.py` | PyRadiomics usage example with empty-string placeholder paths; not runnable as-is, kept only as an API usage reference. |
| `ellipse_inner_cvxpy.py` | Original CVXPY-based inner-ellipsoid fit. Superseded by `scripts/inner_ellipsoid.py` (rewritten using `sklearn`/`ConvexHull`). |
| `lownerjohn_ellipsoid.py` | Original Löwner-John outer-ellipsoid fit. Requires `mosek` (commercial, licensed solver) and `polytope`, neither in requirements.txt. Superseded by `scripts/ellipsoid_inner_outer.py`, which reimplements this without MOSEK. |
| `nestle_ellipsoid.py` | Alternate nested-sampling ellipsoid fit using the `nestle` package (not in requirements.txt). Experimental, never wired into the pipeline. |
| `generation_theoretical_margins.py` | Standalone margin-theory analysis. Predecessor of `scripts/interpolation_volumes_tumor_size_marker.py`. |
| `interpolation_volumes_plot_double_energy_axis.py` | Superset of `scripts/interpolation_volumes_tumor_size_marker.py` with double-energy-axis plotting; trimmed down into the current version. |
| `pie_chart_scatter_plot_lateral_error.py` | Predecessor of `scripts/pie_chart_scatter_plot.py`, includes an extra EAV/PAV ratio and Dice/VOE branch not carried forward. |
| `DistancesVolumes_twinAxes.py` | Standalone twin-axis boxplot script, retired; no current-tree equivalent. |

`E_radiomics_stats_all.py` was removed entirely (not archived): it had an
actual `SyntaxError` (would not even import) and was already fully
superseded by `E_radiomics_stats.py`.
