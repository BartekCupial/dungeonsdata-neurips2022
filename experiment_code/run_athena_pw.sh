#!/usr/bin/env bash

#SBATCH -o logger-%j.txt
#SBATCH --gres gpu:1  
#SBATCH --nodes 1
#SBATCH --mem=20G
#SBATCH -c 20
#SBATCH -t 2880
#SBATCH -A plgcrlgpu-gpu-a100
#SBATCH -p plgrid-gpu-a100


singularity exec --nv \
    -H /net/tscratch/people/plgmbortkiewicz/repos/dungeonsdata-neurips2022/experiment_code \
    --env WANDB_API_KEY=3307fa1432be95ad74806558208e9c2af21a43d5 \
    --env WANDBPWD=$PWD \
    -B /net/pr2/projects/plgrid/plgg_pw_crl/mbortkiewicz/nle:/nle \
    -B $TMPDIR:/tmp \
    /net/pr2/projects/plgrid/plgggmum_crl/bcupial/dungeons.sif \
    ./train.sh
