#!/bin/bash

#SBATCH -p compute    # which partition to run on ('compute' is default)

#SBATCH -J  t12   # arbitrary name for the job (you choose)

# Request gpus
#SBATCH --gres=gpu:1

# specify number of tasks/cores per node required
#SBATCH --ntasks-per-node=8

# SBATCH --mem=30000    # how much RAM you need (30GB in this case), if different from default; your job won't be able to use more than this

#SBATCH -t 2-00:00:00    # maximum execution time: in this case one day, two hours and thirty minutes (optional)

# set to email at start,end and failed jobs
#SBATCH --mail-type=ALL
#SBATCH --mail-user=riju.das@adaptcentre.ie

# Uncomment the following to get a log of memory usage; NOTE don't use this if you plan to run multiple processes in your job and you are placing "wait" at the end of the job file, else Slurm won't be able to tell when your job is completed!

# vmstat -S M {interval_secs} >> memory_usage_$SLURM_JOBID.log &



# Load Conda
source /home/rdas/anaconda3/etc/profile.d/conda.sh

# Activate your Conda environment
conda activate VA_pytorch

# Your commands here
python /home/rdas/student_engagement/train_files/train_hyper_EH-1.py