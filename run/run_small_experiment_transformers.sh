#!/usr/bin/env bash

# Run this script from the project root dir.

environment_setup="
module load Anaconda3
module load CUDA/12.0.0
source activate LGI-env
"

function run_repeats {
    dataset=$1
    method=$2
    # The cmd line cfg overrides that will be passed to the main.py,
    # e.g. 'name_tag test01 gnn.layer_type gcnconv'
    cfg_overrides=$3

    cfg_file="${cfg_dir}/${dataset}/${method}.yaml"
    if [[ ! -f "$cfg_file" ]]; then
        echo "WARNING: Config does not exist: $cfg_file"
        echo "SKIPPING!"
        return 1
    fi

    main="$DATA/.conda/envs/exphormer/bin/python main.py --cfg ${cfg_file}"
    out_dir="results/${dataset}"  # <-- Set the output dir.
    common_params="out_dir ${out_dir} ${cfg_overrides}"

    echo "Run program: ${main}"
    echo "  output dir: ${out_dir}"

    time=`date +%m.%d-%H:%M`
    # Run each repeat as a separate job
    for SEED in {0..4}; do
        echo job name ${method}-${dataset}: seed ${SEED}
        echo ${main} --repeat 1 seed ${SEED} ${common_params}

        sbatch <<EOT
#!/bin/bash
#SBATCH --output="output/${time}-%x-%j.out"
#SBATCH --error="output/${time}-%x-%j.out"
${slurm_directive}
#SBATCH --job-name=${method}-${dataset}-${SEED}
${environment_setup}
${main} --repeat 1 seed ${SEED} ${common_params}
EOT
    done
}


function run_transformer_models {
    dataset=$1
    run_repeats ${dataset} Exphormer
    run_repeats ${dataset} GPS+Transformer
    run_repeats ${dataset} GPS+Performer
    run_repeats ${dataset} GPS+SparseAttention
}



echo "Do you wish to sbatch jobs? Assuming this is the project root dir: `pwd`"
select yn in "Yes" "No"; do
    case $yn in
        Yes ) break;;
        No ) exit;;
    esac
done

################################################################################
##### GPS
################################################################################

# Comment-out runs that you don't want to submit.
cfg_dir="configs/Small-experiment"
slurm_directive="
#SBATCH --partition=short
#SBATCH --time=12:00:00
#SBATCH --mem=60G
#SBATCH --gres=gpu:1
"


run_transformer_models Cifar10


run_transformer_models MalNet-Tiny


run_transformer_models PascalVOC-SP


run_transformer_models Peptides-Func