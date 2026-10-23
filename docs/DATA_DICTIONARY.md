# Data dictionary and provenance

## Reference dataset

`datasets/module30_single_straight_v1/metadata.json` and the CSV manifest define the 200 straight single-crack reference meshes and whole-mesh split (120 training, 40 validation, 40 test). Preserve mesh-level splits when reconstructing training inputs. NPZ member names, dimensions and dtypes are recorded for an inspected sample in `docs/INSPECTED_NPZ_SCHEMA.json`.

Coordinates and mesh sizes use meters. Triangle and edge arrays contain zero-based node indices. Array indices are dimensionless. Geometry is two-dimensional; finite-element displacement vectors contain horizontal and vertical degrees of freedom.

Mesh-size labels are derived from equivalent equilateral triangle sizes h_e=sqrt(4 A_e/sqrt(3)), then aggregated at nodes. The local feature preparation step reuses the stored size labels and area weights and constructs 14 local geometric features. Target sizes are transformed to a normalized logarithmic scale. The size network learns reference mesh sizing; no mechanical solution is required before online size prediction.

GNN labels indicate whether each candidate edge belongs to the reference triangle-edge set. Candidate features have 10 components, while Line Graph adjacency links candidate edges sharing an endpoint. These are not the 14 features used by the size network.

## Mesh and mechanics outputs

The public package retains summary metrics. Full generated mesh and field arrays are produced by running the supplied workflow rather than included for every archived case.

Online geometric NPZ files contain node coordinates and triangle connectivity. Export for mechanical computation duplicates crack-face nodes to represent displacement discontinuities. Geometric node counts and mechanical node counts therefore need not agree.

SI units: coordinates/displacements m; material modulus and stress Pa; strain dimensionless; thickness m; top-edge line load N/m; equivalent node force N; strain energy J. Plotting routines can convert stress to MPa for display without changing raw arrays.

CSV error fields ending in `relative_l2` and relative area errors are dimensionless ratios (multiply by 100 for percent). Angles with suffix `_deg` use degrees. Timing fields ending in `_seconds` use seconds; manuscript forward times in ms require a factor of 1000.

## Archived evidence

Files in `evidence/paper_results/` are copied from existing completed runs. Their numeric values and original provenance fields are retained; any absolute paths inside historical reports describe the original run, not a required runtime location. New outputs belong under ignored `results/` directories.

Older mesh-suite summaries report the original implementation time. The speed benchmark has a different, paired measurement protocol; use its own summary for optimized timing. Do not mix or sum inclusive parent and child timing fields.
