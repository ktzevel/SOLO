# Code for Instance-reference cross-bilateral filter

## Note
This code is adapted from the [fog_simulation_DBF](https://github.com/sakaridis/fog_simulation_DBF) repository.


## Contents:
- Dependencies for running instance-reference cross-bilateral filter adapted from the original repository.
  - `Color_transformations`
  - `Dark_channel_prior`
  - `Depth_processing`
  - `external`
  - `Fog_simulation`
  - `Instance-level_semantic_segmentation`
  - `utilities`
- `filter_acdc_depth.m` (adapted Matlab function for performing the instance-reference cross-bilateral filter.)
- Conversion scripts between .mat and .npy files:
  - `mat2npy.py`
  - `npy2mat.py`
- `main.sh` (example script for running instance-reference cross-bilateral filter)
  - input:
    - instances of the ACDC dataset (sample available here: [instance_label_ids-single_channel.tar.gz](https://mega.nz/file/1rRl0IaS#ebPnE6TJptrD5fbB5G2QehF4G-lkebgQ_vNb-wcPGt0))
    - semantics of the ACDC dataset (sample available here: [acdc_sem_label_ids.tar.gz](https://mega.nz/file/V2JH0SJB#kjLMWENNBQEXqAFL7pW-k14IpEJ5MErRBO3cFoCxQwU))
    - daytime input images (sample available here: []())
    - initial depth estimations in .mat format.
  - outputs:
	- the filtered depth estimations.