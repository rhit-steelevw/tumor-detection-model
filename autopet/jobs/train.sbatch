#!/bin/bash

#SBATCH --job-name=autopet-train

#SBATCH --partition=gpu

#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=8

#SBATCH --mem=96G

#SBATCH --time=24:00:00

#SBATCH --output=%x_%j.out

set -eo pipefail

source /home/steelevw/miniconda3/etc/profile.d/conda.sh

conda activate autopet

export AP=/home/steelevw/18/autopet

export nnUNet_raw=$AP/nnUNet_raw

export nnUNet_preprocessed=$AP/nnUNet_preprocessed

export nnUNet_results=$AP/nnUNet_results

export PRETRAINED=/home/steelevw/18/autopet/pretrained/Dataset619_nativemultistem/MultiTalent_trainer_multistems_4000ep__nnUNetResEncUNetL1x1x1_Plans_znorm_bs24__3d_fullres/fold_all/checkpoint_final.pth

export nnUNet_n_proc_DA=$((SLURM_CPUS_PER_TASK - 2))

export nnUNet_compile=f

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

FOLD=$nnUNet_results/Dataset200_autoPET_lesions/autoPET3_Trainer__nnUNetResEncUNetLPlansMultiTalent__3d_fullres/fold_0

if   [ -f "$FOLD/checkpoint_final.pth" ];  then echo "Training already finished."; exit 0

elif [ -f "$FOLD/checkpoint_latest.pth" ]; then EXTRA=(--c)

else EXTRA=(-pretrained_weights "/home/steelevw/18/autopet/pretrained/Dataset619_nativemultistem/MultiTalent_trainer_multistems_4000ep__nnUNetResEncUNetL1x1x1_Plans_znorm_bs24__3d_fullres/fold_all/checkpoint_final.pth"); fi

echo "args=${EXTRA[*]}"

ls -l "$PRETRAINED"

nvidia-smi

nnUNetv2_train 200 3d_fullres 0 -tr autoPET3_Trainer -p nnUNetResEncUNetLPlansMultiTalent "${EXTRA[@]}"
