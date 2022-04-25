#!/bin/bash
#SBATCH --account=six@cpu
#SBATCH --job-name=annotate_langid_crawl
#SBATCH --partition=cpu_p1
#SBATCH --cpus-per-task=1
#SBATCH --output=res%A_%a
#SBATCH --time=20:00:00

echo "Running job on $hostname"

# load conda environment
source $six_ALL_CCFRWORK/start-prod
conda activate hugo

python /gpfswork/rech/six/urd43gx/code/filtering_crawl/annotate_langid_crawl/annotate_langid_crawl.py ${SLURM_ARRAY_TASK_ID}
