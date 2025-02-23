#!/bin/bash

mat2npy="mat2npy.py"

img_dir="datasets/acdc_rgb"
inst_dir="datasets/instance_label_ids"
sem_dir="datasets/acdc_sem_label_ids"

depth_dir="datasets/depth/UNI-CONV-BILINEAR/mat"
out_dir="datasets/depth/CBF_instances/UNI-CONV-BILINEAR/ss5_lambda0"
mkdir -p $out_dir

matlab -nodesktop -nodisplay -nosplash -r "filter_acdc_depth($img_dir, $inst_dir, $depth_dir, $out_dir); exit;"

# convert to .npy
python ${mat2npy} -d $out_dir -n res_depth -t single
