#!/bin/bash

#SBATCH --job-name=genai_omnilearn_fine_tune
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --account=m3246
##SBATCH --volume="/pscratch/sd/c/ccardona:/pscratch/sd/c/ccardona"
##SBATCH  --image=docker:vmikuni/pytorch:ngc-23.12-v0

##srun  shifter python scripts/train_jetnet.py --local --layer_scale --dataset jetnet30 --fine_tune 
module load conda
conda activate omnilearned
srun python src/omnilearned/train.py --save_tag test --dataset G4 --path /pscratch/sd/c/ccardona/datasets --num_classes 7