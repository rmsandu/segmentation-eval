#!/bin/bash

# You must specify a valid email address!
#SBATCH --mail-user=!!!ADD MAIL ADRESS!!!

# Mail on NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --mail-type=none

# Job name
#SBATCH --job-name="Calculate metrics"

# Runtime and memory
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=16G

# Partition
#SBATCH --partition=all

#### Your shell commands below this line ####
module load Python/3.6.4-foss-2018a

source ~/envs/maverric/bin/activate

cd ~/src/segmentation-eval

python A_read_files_info.py  -o /storage/research/artorg_igt/Projects/MAVERRIC/output -b /storage/research/artorg_igt/Projects/MAVERRIC/batch.xlsx

