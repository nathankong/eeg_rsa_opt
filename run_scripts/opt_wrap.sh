#!/bin/bash

declare -a LAYERS=( "pool1" "pool2" "pool3" "pool4" "pool5" "fc2" )
declare -a DISTANCES=( "euclidean" "pearson" "decoding" )

model="linearcombo"
#model="elementwise"

script="${GROUP_HOME}/nclkong/eeg_neuralnetwork_rsa/run_scripts/optimize_linear_combo.sh"

for layer in ${LAYERS[@]};
do
    if [[ $layer == fc2 ]]; then
        t=4:00:00
    elif [[ $layer == pool5 ]]; then
        t=6:00:00
    elif [[ $layer == pool4 ]]; then
        t=12:00:00
    elif [[ $layer == pool3 ]]; then
        t=12:00:00
    elif [[ $layer == pool2 ]]; then
        t=18:00:00
    elif [[ $layer == pool1 ]]; then
        t=18:00:00
    else
        echo "Wrong layer."
        exit
    fi

    if [[ $model == elementwise ]]; then
        t=1:00:00
    fi

    for dist in ${DISTANCES[@]};
    do
        jn="${layer}_${dist}_${model}"
        out="outs/${layer}_${dist}_${model}.out"
        err="errs/${layer}_${dist}_${model}.err"

        sbatch -p hns,gpu \
            --job-name=$jn \
            --output=$out \
            --error=$err \
            --nodes=1 \
            --mail-type=FAIL \
            --mail-user=nclkong@stanford.edu \
            --gpus=1 \
            --time=$t \
            --mem=50G \
            --wrap="bash $script $layer $dist $model"
        sleep 0.5

        echo "${layer}, ${dist}, ${model}"
    done
done

