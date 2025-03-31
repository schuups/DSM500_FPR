# DSM500 regarding NVIDIA Modulus's GraphCast PyTorch re-implementation

This repository contains a streamlined version of the code, scripts and results from the DSM500 project execution, providing clear, concise insights and reproducibility information - that matches and supports the Final Project Report reading flow. Given the refactoring, it might be that in the movement of files from the old repository some file paths might have not been updated. This might be the case in particular for notebooks. Nonetheless, the logic is still valid.


## Key folders

| Folder | Contents description |
|--------|-------------|
| [fourcastnet-92260c1](./fourcastnet-92260c1) | The FourCastNet codebase used to train the external reference model (commit 92260c1) |
| [modulus-a5275d8](./modulus-a5275d8) | The codebase copied from the official NVIDIA Modulus github repository (commit a5275d8) |
| [modulus-baseline](./modulus-baseline) | The refactored version of GraphCast used as a baseline for performance improvements. |
| [modulus-dry-run](./modulus-dry-run) | The version of Modulus GraphCast used to collect the dry-run results |
| [modulus-improved](./modulus-improved) | The version of Modulus GraphCast on which the improvements have been implemented. |
| [supplementary_files](./supplementary_files) | Contains supporting code and scripts, e.g. used to generate dataset means and std values needed for normalisation, or to perform inference on a trained model. |
