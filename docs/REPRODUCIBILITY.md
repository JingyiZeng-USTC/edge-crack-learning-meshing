# Reproducibility map

## Core workflow

`run_full_pipeline.py` orchestrates mesh generation, crack-face export and the active FCNN-SM solver. `gnn/23_fast_mesh_entry.py` installs scalar geometry and interpolation kernels via the profiling entry before invoking `gnn/10_learned_size_to_gnn.py`. The fast kernels preserve the candidate-selection rules; small CUDA probability differences are evaluated separately from final mesh identity.

`run_training.py` prepares features and trains the size-field and structured GNN models from the bundled reference data. Offline re-training is optional for inference using the bundled checkpoints. Dataset preparation and training graphs require additional disk space.

## Paper result files

- `evidence/paper_results/size_field/`: saved size-network test metrics and training history.
- `evidence/paper_results/gnn/`: held-out classification metrics and structured recovery summaries.
- `evidence/paper_results/fixed_nodes/`: 40-case fixed-node Gmsh/GNN comparison, including per-case results when available.
- `evidence/paper_results/end_to_end/`: mesh-suite and neural mechanics summaries for 12 straight and 4 curved cases, with original per-case metrics.
- `evidence/paper_results/size_comparison/`: rule-based Gmsh sizing versus learned sizing convergence data, configuration and summary.
- `evidence/mesh_speed_benchmark/`: paired timing records copied from the final workflow package.

## Optional fixed-node comparison

`gnn/20_compare_fixed_nodes_gnn_gmsh.py` reconstructs connectivity on Gmsh's frozen final nodes. It imports `gnn/gmsh_local_refinement_comparison/common_pipeline.py`. Candidate k defaults to 20. Its default test-list path is supplied in `gnn/artifacts/single_straight_200_structured_decoder_recovery_k20/per_mesh_metrics.csv`. Local quality passes are disabled and node coordinate equality is checked after decoding. A single-node embedded crack line is used for this planar connectivity experiment; it is not the crack-face displacement representation of the mechanical solver.

## Verification performed during release preparation

Syntax parsing, active module imports, checkpoint dimensions, file checksums, reference split counts and citation-file parsing are checked. Original numerical evidence is copied without claiming new model training or rerunning all numerical experiments. Packaging smoke-test results do not constitute paper validation.

Exact run times depend on hardware, drivers and software builds. The archived FCNN solver errors compare against direct solutions on the same mesh; the mesh-size convergence experiment uses its separately configured reference solution.
