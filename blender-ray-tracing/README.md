# Code for Ray Tracing


## Note
This script only works for `Blender 3.4.1`. For any other versions of Blender modifications are necessary.


## Input
1. reconstructed meshes (.obj format)
2. albedo estimations (sample available here: [albedo-BS1-1920x1080-joint_bilateral-sRGB-TIFF.tar.gz](https://mega.nz/file/FiZWEazL#D7ZsDRLLLvAWiFYQkyEHPXEdviWMuqowmNzUFlRMuIs))
3. roughness estimations (sample available here: [rough-BS1-1920x1080-joint_bilateral-sRGB-TIFF.tar.gz](https://mega.nz/file/F6wg2IqD#VwX0p28YF8Z2RcezWOmQ37h5R0AZdrGnZInkFZmJFGc))
4. instantiated light sources (sample available here: [light_masks-s1432-240201.tar.gz](https://mega.nz/file/ErBUjD7R#OYBivkvdVXizRkxjzf-Jo5lCq6PfmSXnXZGTrL0N0pE)
   
## Output
- The linear noise-free nighttime images.