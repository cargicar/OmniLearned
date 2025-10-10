#!/bin/bash

#SBATCH --job-name=Dit_shapenet
#SBATCH --nodes=1
#SBATCH --time=10:00:00
#SBATCH --constraint=gpu
#SBATCH --gpus=4
#SBATCH --qos=regular
#SBATCH --account=m3246
##SBATCH --volume="/pscratch/sd/c/ccardona:/pscratch/sd/c/ccardona"
##SBATCH  --image=docker:vmikuni/pytorch:ngc-23.12-v0

##srun  shifter python scripts/train_jetnet.py --local --layer_scale --dataset jetnet30 --fine_tune 
module load conda
conda activate omnilearned
module load pytorch
#srun torchrun --nproc_per_node=4 scripts/train_DiT.py 
srun torchrun --nproc_per_node=4 scripts/train_calopodit.py --dataset ShapeNetCore --num_classes 4 --gap_classes 0 --no_energy_cond --out_channels 3 --in_features 3 --max_particles 2000

