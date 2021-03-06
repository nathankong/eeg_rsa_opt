#!/bin/bash

module load py-scipystack
module load py-scikit-learn
module load py-pytorch/1.0.0_py27

layer=$1
dist=$2
model=$3
lr=0.001
eps=0.001
latentdim=1000
lfpath="${GROUP_HOME}/nclkong/eeg_neuralnetwork_rsa/layer_features/"
rdms_dir="${GROUP_HOME}/nclkong/eeg_neuralnetwork_rsa/base_results_dir/rdms/kaneshiro/"
optim_results_dir="${GROUP_HOME}/nclkong/eeg_neuralnetwork_rsa/base_optim_results_dir/"

echo "Layer: " $layer
echo "Dist: " $dist
echo "Learning rate: " $lr
echo "Epsilon: " $eps
echo "Latent dim: " $latentdim
echo "Model: " $model

python ${GROUP_HOME}/nclkong/eeg_neuralnetwork_rsa/compute_optimal_correlation_flex.py --dataset kaneshiro --distance $dist --model $model --rdms-dir $rdms_dir --nfolds 4 --layer $layer --results-dir $optim_results_dir --layer-features-path $lfpath --latentdim $latentdim --lr $lr --eps $eps

