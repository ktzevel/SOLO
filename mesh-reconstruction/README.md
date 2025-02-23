# Code for Mesh Reconstruction and Depth Refinement


## Contents:
- `calibration.py` (Contains the calibration matrix information specific to GoPro4 which was used to capture the ACDC dataset used in this work.)
- `depth_refinement.py` (The implementation of the Normal-guided depth refinement process.)
- `get_mesh.py` (The implementation of the back-projection that takes as input a refined depth map and outputs a reconstructed mesh.)
- `handle_intersection.py` (Depth post-processing utility.)
- `uncertain.py` (The implementation of uncertainty maps used in the reconstruction process.)
- `post_process_avg.py` (Mesh post-processing utility.)
- `options.py` (The hyper-parameters for the normal-guided depth refinement process.)
- `main.py` (The main script that uses the aforementioned python modules and generates a reconstructed mesh.)

## Input:
- instances of the ACDC dataset (sample available here: [instance_label_ids-single_channel.tar.gz](https://mega.nz/file/1rRl0IaS#ebPnE6TJptrD5fbB5G2QehF4G-lkebgQ_vNb-wcPGt0))
- semantics of the ACDC dataset (sample available here: [acdc_sem_label_ids.tar.gz](https://mega.nz/file/V2JH0SJB#kjLMWENNBQEXqAFL7pW-k14IpEJ5MErRBO3cFoCxQwU)	)
- filtered depth estimations (sample available here: [unidepth_conv-bilinear_upsampled-bilateral_filter_ss5_l0.tar.gz](https://mega.nz/file/9npRAIjB#8ClDstsWaC2fcVROnk3pZbUVm8jmNgLAM7vvhv3eW0g))
- surface normal estimations (sample available here: [idisc-trained_on_nyu-acdc-normals.tar.gz](https://mega.nz/file/1zQSnbZT#YR1L_gTzdkZz5F2ixh7G3bJ8rzjZwgzkVC49riAOVAU))
- daytime input images from reference split of nighttime ([download from ACDC dataset website](https://acdc.vision.ee.ethz.ch/download))


## Output:
- A directory containing the reconstructed meshes in .obj format.
- The refined depth from the normal-guided depth refinement process.