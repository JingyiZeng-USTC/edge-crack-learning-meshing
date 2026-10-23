# Learning-based adaptive meshing for edge-crack stress analysis

Research software and supporting data for **Learning-based adaptive meshing with graph neural network-based connectivity prediction for edge-crack stress analysis**.

Authors: Jingyi Zeng and Guang Wu.

Repository: https://github.com/1330519885zjy-coder/edge-crack-learning-meshing

This repository couples a learned absolute mesh-size field, adaptive Poisson sampling, Line Graph GraphSAGE connectivity prediction, constraint-aware mesh recovery, and a mesh-specific FCNN-SM displacement solver. Strain and stress follow from constant-strain triangular finite elements. Validation concerns two-dimensional, single, non-branching edge cracks under small-deformation linear elasticity, including straight and curved paths.

## Contents

- `module30_learned_size_field/`: size targets, 14 geometric input features, network training and continuous size-field inference.
- `sampling/`: boundary and crack sampling, auxiliary points and adaptive interior sampling.
- `gnn/`: candidate edges, line graphs, GraphSAGE training, structured decoding and mesh-quality improvement.
- `module10_fem/`: stiffness assembly, line loads, displacement constraints and field recovery.
- `module44_load_surrogate_fcnn_sm/`: the active load-to-displacement neural solver.
- `module43_coordinate_fcnn_sm/`: compatibility utilities used for result export and comparison by the active solver.
- `datasets/module30_single_straight_v1/`: 200 reference meshes, labels and the original 120/40/40 split.
- `module30_checkpoints/` and `gnn/artifacts/`: pretrained size and connectivity models.
- `evidence/`: archived per-case summary metrics and benchmark records.
- `docs/`: data definitions, reproduction map and release status.
- `CITATION.cff`: software citation metadata.
- `MANIFEST_SHA256.json`: file integrity inventory.

## Environment

Python 3.10 was used for the archived experiments. Package versions are recorded in `environment_versions.json`. PyTorch and PyTorch Geometric must be installed for a mutually compatible CPU/CUDA environment. `requirements.txt` lists the workflow dependencies; `requirements-comparison.txt` adds Gmsh and Pillow for comparison experiments. No commercial finite-element software is required for the main pipeline.

```bash
python -m pip install -r requirements.txt
python -B verify_bundle.py
```

Install the appropriate PyTorch build before the remaining dependencies when CUDA is required. The recorded environment is a provenance record, not a guarantee that every old wheel remains available on every platform.

## End-to-end computation

```bash
python -B run_full_pipeline.py --case-name straight_demo
python -B run_full_pipeline.py --geometry-json examples/curved_left_mid_up.json --case-name curved_demo
python -B run_full_pipeline.py --suite
```

Defaults: W=1 m, H=2 m, thickness=0.01 m, E=2.1e11 Pa, Poisson ratio=0.3, plane stress, and a uniform upward top-edge line load of 5e5 N/m. Bottom vertical displacements and one bottom horizontal displacement are constrained. Nodes are generated using the learned size field; online GNN candidates use k=20. The quality target is 20 degrees and attainment is checked numerically.

The size-field and GNN checkpoints are reused online. FCNN-SM is optimized separately on every new mesh, with no displacement or stress labels. Its final output is not corrected using a direct finite-element solve or PCG. A same-mesh direct finite-element solution is subsequently computed for evaluation.

Online mesh recovery uses scored candidate edges and geometry checks; it does not call CDT/Delaunay/Triangle as a triangulation fallback. Some compatibility files retain older meshing interfaces that are outside this active call path. Crack-face node duplication creates independent displacement degrees of freedom after geometric mesh generation.

## Offline training

```bash
python -B run_training.py --stage size --size-epochs 300
python -B run_training.py --stage gnn --gnn-epochs 80
```

The historical GNN training entry defaults to k=16, whereas final online inference uses k=20. Changing the training setting constitutes a new experiment. Both offline models use the original reference meshes; training nodes are not resampled by the learned field. The FCNN variable named `epoch` denotes one random-minibatch optimizer update.

## Results and interpretation

Archived results include size prediction, held-out GNN evaluation, fixed-node GNN/Gmsh connectivity comparison, 12 straight plus 4 curved end-to-end cases, and the mesh-speed benchmark. See `docs/REPRODUCIBILITY.md` for files and experiment-specific settings. Archived outputs are distinguished from new computations, and the 2-step packaging smoke test is not presented as a paper accuracy result.

The fixed-node comparison freezes Gmsh's final nodes, supplies identical coordinates and constraints to the GNN branch, and disables quality edge flips and node movement in that branch. It evaluates connectivity separately from learned node placement. The neural displacement error is measured against the direct finite-element solution on the same mesh; this is a solver-accuracy comparison, not a mesh-convergence reference.

## Citation and access

Use the authors, title and repository URL in `CITATION.cff`. An archival DOI can be added to this metadata when a versioned deposit is available.

## Licenses

Author-owned source code is licensed under MIT (`LICENSE`). Author-owned datasets, pretrained model parameters and archived numerical results are licensed under CC BY 4.0 (`LICENSE-DATA.md`). Third-party packages and third-party materials retain their own licenses. See `THIRD_PARTY_NOTICES.md`.
