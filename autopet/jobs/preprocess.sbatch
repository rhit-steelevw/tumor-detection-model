#!/bin/bash

#SBATCH --job-name=autopet-prep

#SBATCH --partition=cpu

#SBATCH --cpus-per-task=16

#SBATCH --mem=128G

#SBATCH --time=24:00:00

#SBATCH --output=%x_%j.out

set -xeo pipefail

source /home/steelevw/miniconda3/etc/profile.d/conda.sh

conda activate autopet

export AP=/home/steelevw/18/autopet

export AP3=/home/steelevw/autopet-3-submission

export nnUNet_raw=$AP/nnUNet_raw

export nnUNet_preprocessed=$AP/nnUNet_preprocessed

export nnUNet_results=$AP/nnUNet_results

mkdir -p $nnUNet_preprocessed $nnUNet_results

which nnUNetv2_plan_and_preprocess

echo "nnUNet_raw=$nnUNet_raw"

N=$SLURM_CPUS_PER_TASK

nnUNetv2_plan_and_preprocess -d 200 -c 3d_fullres -np $N -npfp $N

nnUNetv2_plan_and_preprocess -d 201 -c 3d_fullres -np $N -npfp $N

cp $AP/splits_final.json $nnUNet_preprocessed/Dataset200_autoPET_lesions/

nnUNetv2_merge_lesion_and_organ_dataset -l 200 -o 201

python $AP/copy_plans.py

echo "PREPROCESSING DONE"
