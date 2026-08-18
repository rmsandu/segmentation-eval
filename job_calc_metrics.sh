#!/bin/bash

# You must specify a valid email address!
#SBATCH --mail-user=<you@example.org>

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
# Fill in these environment variables (or export them before submitting the job)
# before running: SEGEVAL_ENV_DIR, SEGEVAL_SRC_DIR, SEGEVAL_OUTPUT_DIR, SEGEVAL_BATCH_XLSX
source "${SEGEVAL_ENV_DIR:?set SEGEVAL_ENV_DIR to your virtualenv path}/bin/activate"

cd "${SEGEVAL_SRC_DIR:?set SEGEVAL_SRC_DIR to the segmentation-eval checkout path}"

python A_read_files_info.py -o "${SEGEVAL_OUTPUT_DIR:?set SEGEVAL_OUTPUT_DIR}" -b "${SEGEVAL_BATCH_XLSX:?set SEGEVAL_BATCH_XLSX}"
