#!/bin/bash

#SBATCH --job-name="segf_b4"

#----------- TO CHANGE -----------
#SBATCH --output="logs/segf_b4-%j.out"
#---------------------------------

#SBATCH --time=120:00:00
#SBATCH --gpus=1
#SBATCH --gres=gpumem:39g
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=8
#SBATCH --tmp=80000

#----------------------------------------------------------------------------------------------------
#
# Datasets config: configs/_base_/datasets/light.py # dataloaders, workers, etc.
# LightDataset class is located in: mmseg/datasets/light.py # classes names, files prefixes, etc.
#
# model configs define weights, number of classes, etc.
# - configs/segformer/segformer_mit-b5_8xb1-160k_light-1024x1024.py (transformer based)
# - configs/hrnet/hrnet_mit-b5_8xb1-160k_light-1024x1024.py (convolution based)
# - configs/_base_/models/segformer_mit-b0.py
# - configs/_base_/models/fcn_hr18.py (num_classes)
#----------------------------------------------------------------------------------------------------

REPO="/cluster/home/ktzevelekaki/workspace/night_simulation/mmsegmentation"
export PYTHONPATH="$REPO:$PYTHONPATH" # needed for custom registered modules like LightDataset class.

#----------- TO CHANGE -----------
CONFIG="${REPO}/configs/segformer/segformer_mit-b4_8xb1-160k_light-1024x1024.py"
#---------------------------------

PYENV="${HOME}/venvs/mmseg_new"
MODULES="${HOME}/modules/mmseg_new"
source ${MODULES}
source ${PYENV}/bin/activate

# logs & checkpoints
EXP=$(echo $(basename $CONFIG) | cut -d. -f1)
TMP_RESDIR="${SCRATCH}/lights_ann_net/${EXP}"
mkdir -p $TMP_RESDIR

# training & testing data
DATADIR=${TMPDIR}

# extracts data to TMPDIR
#----------- TO CHANGE -----------
tar -I pigz -xvf ${SCRATCH}/datasets/light_sources_annotations_dataset.tar.gz -C $DATADIR
# if the number of categories changes, you need to change the dataset's annotations
# 
#---------------------------------

cd $REPO
DATA_ROOT=${DATADIR}/light_sources_annotations_dataset
python -u tools/train.py ${CONFIG} --work-dir=$TMP_RESDIR \
		--cfg-options data_root="$DATA_ROOT" \
					  train_dataloader.dataset.data_root="$DATA_ROOT" \
					  val_dataloader.dataset.data_root="$DATA_ROOT" \
					  test_dataloader.dataset.data_root="$DATA_ROOT"

DATE=$(date '+%y_%m_%d_%H_%M')
RESDIR="/cluster/work/cvl/ktzevelekaki/${EXP}-${DATE}"
mkdir -p $RESDIR

# copy important config files to the results folder.
SCRIPT_DIR="$( cd "$( dirname "${0}" )" && pwd )"
SCRIPT_FN="$(basename "${0}")"

cp configs/_base_/datasets/light.py $TMP_RESDIR
cp mmseg/datasets/light.py $TMP_RESDIR
cp ${SCRIPT_DIR}/${SCRIPT_FN} $TMP_RESDIR # sbatch script as well.

# Copy to a permanent storage space.
cp -r $TMP_RESDIR $RESDIR && rm -r $TMP_RESDIR
