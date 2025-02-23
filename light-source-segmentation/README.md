# Code for the light source segmentation (ACDC Light Sources)


## Note:
The code is adapted from the [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) repo.

## Contents:
- `slurm/` (Contains slurm sbatch script for `training` segmentation models - `segformer` or `hrnet` - on the light source segmentation task.)
- `generate_light_annotations.sh` (The main script used to compile the ACDC light source dataset.)
  - Inputs:
	- Manually annotated light source dataset (semantics & instances)
	- Pre-trained light source model (config file, iteration and checkpoint)

  - Outputs:
	- instance
	- semantic
	- panoptic level annotations

  - Steps:
	1. Inference of the semantic level annotations for the unlabeled acdc reference images.
	2. Conversion to instances from the aforementioned semantic level annotations.
	3. Merging of the estimated and manually created annotations.
	4. Traffic lights clustering (replacement of the class traffic_light with traffic_light_ + {red, orange, green})
	5. Conversion to panoptic level annotations.
  
  - Checkpoint:
    - A checkpoint of a trained light source segmentation model can be found here: [segformer_mit-b5_8xb1-160k_light-1024x1024-iter_128000.pth](https://mega.nz/file/wqh3FbjQ#EPlJZPTdX5qHD5M1JgHJEJXe-75BHaLkVXngJWvfJTM)


- `traffic_light_clustering.py` (This script implement the traffic light clustering using the k-means clustering algorithm in the lab color space.)

- `light_ann_labels.csv` (Is a label index for the light source categories)


## ACDC Light Sources

### Download:
For now, you can download the ACDC light source panoptic annotations from [here](https://mega.nz/file/0jZniLSB#iVc0Qh894VvnMqM_Xj0sTv_M4E97CmdZ62QWIb2vIlM).

