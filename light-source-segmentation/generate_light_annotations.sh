#!/bin/bash

#SBATCH --job-name="light_prep"
#SBATCH --output="logs/light_prep-%j.out"

#SBATCH --time=4:00:00
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=8
#SBATCH --tmp=80000

#---------------------------------------------------------------------------------------
# DESCRIPTION: Generation of the full light source annotation dataset.
#---------------------------------------------------------------------------------------
# Inputs:
# Manually annotated light source dataset (semantics & instances)
# Pre-trained light source model (config file, iteration and checkpoint)
#
# Outputs:
# instance, semantic and panoptic level annotations.
#
# Steps:
# 1. Inference of the semantic level annotations for the unlabeled acdc ref. images.
# 2. Conversion to instances from the aforementioned semantic level annotations.
# 3. Merging of the manually created and infered annotations.
# 4. Traffic lights clustering and update of the semantic annotations introducing 3 new classes for the traffic lights.
# 5. Conversion to panoptic level annotations.
#
#---------------------------------------------------------------------------------------
# EDIT HERE:
#---------------------------------------------------------------------------------------
DATE=$(date +'%y%m%d')

# The 350 manually annotated images.
ANN_DIR="$SCRATCH/datasets/light_sources_annotated"
ANN_SEM_DIR="$ANN_DIR/semantic"
ANN_INST_DIR="$ANN_DIR/instance"

# The 653 infered annotations.
PRED_DIR="$TMPDIR/${DATE}_pred_light_annotations"
PRED_SEM_DIR="$PRED_DIR/semantic"
PRED_INST_DIR="$PRED_DIR/instance"

# The combined manual and infered annotations.
ALL_DIR="$SCRATCH/datasets/light_source_annotations_full_${DATE}"
ALL_SEM_DIR="$ALL_DIR/semantic"
ALL_INST_DIR="$ALL_DIR/instance"
ALL_PAN_DIR="$ALL_DIR/panoptic"

# ACDC's rgb reference images.
IMG_TAR_DIR="${SCRATCH}/datasets/acdc_rgb.tar.gz"
IMG_DIR="$TMPDIR/acdc_rgb"
IMG_DIR_PRED="$TMPDIR/acdc_rgb_pred" # images to predict light annotations.

# The directory of the trained light source annotation network.
MODEL_DIR="${HOME}/cvl_space/light_ann_net/segformer_mit-b5_8xb1-160k_light-1024x1024"
CKPT_FN="iter_128000.pth"

REPO_DIR="$HOME/workspace/night_simulation/mmsegmentation"

traffic_light_clustering="$HOME/workspace/night_simulation/acdc_merge/traffic_light_clustering.py"
acdc2panoptic="$HOME/workspace/night_simulation/acdc_merge/acdc2panoptic.py"
#---------------------------------------------------------------------------------------

# -------------------------------------
echo "Enabling to mmseg_new environment..."
deactivate
module purge

# loads necessary modules.
source $HOME/modules/mmseg_new
# activates the pyenv.
source $HOME/venvs/mmseg_new/bin/activate
# -------------------------------------

# adds the repo's dir in the PYTHONPATH for custom modules.
export PYTHONPATH=PYTHONPATH:$REPO_DIR

CONFIG=${MODEL_DIR}/$(basename $MODEL_DIR).py

echo "Re-links to tmp directory..."
# create the old link name and make it point to the new TMPDIR.
OLD_TMP=$(grep -o "/scratch/tmp[^,]*${USER}" ${CONFIG} | head -n1)
ln -sf $TMPDIR $OLD_TMP


echo "Extracts acdc reference images..."
# extract rgb images.
tar -xf $IMG_TAR_DIR $fids -C $TMPDIR

echo "Finds manually annotated FIDS..."
ANN_FIDS="annotated_fids"
find $ANN_DIR/ -type f | xargs -I@  basename @ | cut -d. -f1 | sort | uniq > $ANN_FIDS

# separate manually annotated ones.
mkdir -p $IMG_DIR_PRED
find $IMG_DIR -type f | grep -vf $ANN_FIDS |xargs -I@ cp @ $IMG_DIR_PRED

mkdir -p $PRED_SEM_DIR
mkdir -p $PRED_INST_DIR

echo "step 1: Performs inference to predict light annotations..."
python tools/infer.py $IMG_DIR_PRED $PRED_SEM_DIR $CONFIG $MODEL_DIR/$CKPT_FN

echo "step 2: Converts semantic predictions to instance ones..."
python tools/semantic2instance.py $PRED_SEM_DIR $PRED_INST_DIR

echo "step 3: Combines predicted and annotated masks..."
mkdir -p $ALL_SEM_DIR 
mkdir -p $ALL_INST_DIR 
cp $ANN_SEM_DIR/* $ALL_SEM_DIR
cp $PRED_SEM_DIR/* $ALL_SEM_DIR

cp $ANN_INST_DIR/* $ALL_INST_DIR
cp $PRED_INST_DIR/* $ALL_INST_DIR

# -------------------------------------
echo "Changing to masks environment..."
deactivate
module purge

# loads necessary modules.
source $HOME/modules/masks
# activates the pyenv.
source $HOME/venvs/masks/bin/activate
# -------------------------------------

echo "step 4: Clusters traffic lights given their rgb colors in the ref. acdc images."
mkdir -p $ALL_SEM_DIR/gt $ALL_SEM_DIR/rgb
cp $IMG_DIR/*.png $ALL_SEM_DIR/rgb
cp $ALL_SEM_DIR/*.png $ALL_SEM_DIR/gt
python $traffic_light_clustering $ALL_SEM_DIR $ALL_SEM_DIR 

# clean auxiliary images
rm -r $ALL_SEM_DIR/gt $ALL_SEM_DIR/rgb

echo "step 5: Converts to panoptic level annotations..."
mkdir -p $ALL_PAN_DIR 
python $acdc2panoptic -s=$ALL_SEM_DIR -i=$ALL_INST_DIR -o=$ALL_PAN_DIR --label_set=ls

echo "Done!"
