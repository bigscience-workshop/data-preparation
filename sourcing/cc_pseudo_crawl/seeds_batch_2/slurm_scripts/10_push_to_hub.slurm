#!/bin/bash
#SBATCH --job-name=pseudo-crawl-push-to-hub               # (change me!) job name
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1                                             # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=4                                               # (change me! between 0 and 48) number of cores per tasks
#SBATCH --hint=nomultithread                                            # we get physical cores not logical
#SBATCH --time 10:00:00                                                 # (change me! between 0 and 20h) maximum execution time (HH:MM:SS)
#SBATCH --gres=gpu:0                                                    # (change me! between 0 and 1) number of gpus
#SBATCH --output=logs/%x-%j.out           # output file name #TODO change path if necessary
#SBATCH --account=six@cpu                                               # account
#SBATCH -p compil                                                       # partition with internet

set -x -e

source $HOME/start-modelling-metadata-user

# mv $six_ALL_CCFRSCRATCH/pseudo_crawl/seeds_batch_2/datasets-compressed-shards/bigscience-catalogue-data/* $six_ALL_CCFRSCRATCH/pseudo_crawl/hub/pseudo_crawl/ not used

rsync -vam -f'+ *.jsonl.gz' -f'+ */' -f'- *' $six_ALL_CCFRSCRATCH/pseudo_crawl/seeds_batch_2/datasets-compressed-shards/bigscience-catalogue-data/* $six_ALL_CCFRSCRATCH/pseudo_crawl/hub/pseudo_crawl/

# ls $six_ALL_CCFRSCRATCH/pseudo_crawl/seeds_batch_2/datasets-compressed-shards/bigscience-catalogue-data/ -f'+ *.jsonl.gz' | xargs -n1 -P8 -I% rsync -Pa % $six_ALL_CCFRSCRATCH/pseudo_crawl/hub/pseudo_crawl/ not used


cd $six_ALL_CCFRSCRATCH/pseudo_crawl/hub/pseudo_crawl/

git status

for seed_id in {698..708}
do
    echo "Add seed id n°$seed_id"
    git add -v *seed_id="$seed_id"/*.gz
done

git commit -v -m "add depth 0 dataset with html content extracted"
git push -v
