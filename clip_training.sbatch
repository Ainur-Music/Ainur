#!/bin/bash
#SBATCH --job-name=ainur_clip_training
#SBATCH --mail-type=ALL
#SBATCH --mail-user=giuseppe.concialdi@studenti.polito.it
#SBATCH --partition=cuda
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=job_name_%j.log
#SBATCH --mem-per-cpu=25000M
#SBATCH --cpus-per-task 8

# Clip training
python ainur/model/clip.py --batch_size 16 --num_workers 8
