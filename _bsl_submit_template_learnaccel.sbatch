#!/bin/bash
#SBATCH --job-name=bsl
#SBATCH --partition=learnaccel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=125G
#SBATCH --gpus-per-node=2
#SBATCH --constraint=volta32gb
#SBATCH --time=3-0:00:00
#SBATCH --chdir=/private/home/xhan77/cad-private

source /private/home/xhan77/.bashrc
conda activate contrast_decode

srun bash bsl_testrun_decode_ar_contrastive_fileio.sh 2023 "0,1" 1e-4 1000 "n/a" $TESTFILE 5 1 1.0 "bsl_test_ensemble" 0 $RTRUNC $GENLEN 0.0 $TOPP "n/a"