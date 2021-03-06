#!/bin/bash 
#
#all commands that start with SBATCH contain commands that are just used by SLURM for scheduling
#################
#set a job name  
#SBATCH --job-name=k_all
#################
#a file for job output, you can check job progress
#SBATCH --output=outs/k_all.out
#################
# a file for errors from the job
#SBATCH --error=errs/k_all.err
#################
#SBATCH --time=24:00:00
#SBATCH -p hns,normal
#################
# --gres will give you one GPU, you can ask for more, up to 8 (or how ever many are on the node/card)
# We are submitting to the gpu partition, if you can submit to the hns partition, change this to -p hns_gpu.
#################
#number of nodes you are requesting
#SBATCH --nodes=1
#################
#memory per node; default is 4000 MB per CPU
#SBATCH --mem=50G
#################
# Have SLURM send you an email when the job ends or fails, careful, the email could end up in your clutter folder
#SBATCH --mail-type=FAIL # notifications for job done & fail
#SBATCH --mail-user=nclkong@stanford.edu

module load py-scipystack
module load py-scikit-learn
module load py-numpy

srun python ${GROUP_HOME}/nclkong/eeg_neuralnetwork_rsa/perform_all_analyses_new.py --dataset kaneshiro --results-dir ${GROUP_HOME}/nclkong/eeg_neuralnetwork_rsa/base_results_dir/ --data-file-path ${GROUP_HOME}/eeg_data/kaneshiro_data/kaneshiro_eeg.npy --layer-rdm-dir ${GROUP_HOME}/nclkong/eeg_neuralnetwork_rsa/layer_rdm_ps/


