## EEG-CNN Comparison

1. Compute EEG RDMs using `perform_all_analyses_new.py` with the arguments set accordingly. 
See `run_scripts/compute_rdms.sbatch`.
2. Optimize linear combinations of VGG19 weights so that the RDM obtained from each model layer maximally correlates with the EEG RDM. 
To do this, run `compute_optimal_correlation_flex.py` with the arguments set accordingly. See `run_scripts/optimize_linear_combo.sh`.

Note that VGG19 features for `pool1`, `pool2`, `pool3`, `pool4`, `pool5`, `fc2` model layers must be pre-computed (with the 72 stimuli)
and saved in `layer_features/`.
