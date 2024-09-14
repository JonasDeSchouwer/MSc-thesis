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

    # --- Do the desired cfg overrides here ---
    cfg_overrides="${cfg_overrides} train.ckpt_best True"

    # --- Set the dataset-specific cfg overrides ---
    # if dataset is ShapeNet-Part
    if [[ $dataset == "ShapeNet-Part" ]]; then
        cfg_overrides="${cfg_overrides} gnn.head inductive_node dataset.format PyG-ShapeNet metric_best f1"
    # elif dataset is ModelNet10
    elif [[ $dataset == "ModelNet10" ]]; then
        cfg_overrides="${cfg_overrides} gnn.head graph dataset.format PyG-ModelNet10 metric_best accuracy"
    # elif dataset is ModelNet40
    elif [[ $dataset == "ModelNet40" ]]; then
        cfg_overrides="${cfg_overrides} gnn.head graph dataset.format PyG-ModelNet40 metric_best accuracy"
    # elif dataset is S3DIS
    elif [[ $dataset == "S3DIS" ]]; then
        cfg_overrides="${cfg_overrides} gnn.head inductive_node dataset.format PyG-S3DISOnDisk metric_best f1"
    fi

    cfg_file="configs/PC/${method}.yaml"
    if [[ ! -f "$cfg_file" ]]; then
        echo "WARNING: Config does not exist: $cfg_file"
        echo "SKIPPING!"
        return 1
    fi

    main="$DATA/.conda/envs/exphormer/bin/python -O main.py --cfg ${cfg_file}"
    out_dir="results/PC/${dataset}"  # <-- Set the output dir.
    common_params="out_dir ${out_dir} ${cfg_overrides}"

    echo "Run program: ${main}"
    echo "  output dir: ${out_dir}"

    time=`date +%m.%d-%H:%M`
    # Run each repeat as a separate job
    for SEED in {0..2}; do  # only 3 runs because my priority credits ran out
        echo job name ${method}-${dataset}: seed ${SEED}
        echo ${main} --repeat 1 seed ${SEED} ${common_params}

        sbatch <<EOT
#!/bin/bash
#SBATCH --output="output/${time}-%x-%j.out"
#SBATCH --error="output/${time}-%x-%j.err"
${slurm_directive}
#SBATCH --job-name=${method}-${dataset}-${SEED}
${environment_setup}
nvidia-smi
lscpu
${main} --repeat 1 seed ${SEED} ${common_params}
EOT
    done
}


function run_gnn_baselines {
    dataset = $1
    for layers in 5 10 15; do
        run_repeats ${dataset} GCN-$layers
        run_repeats ${dataset} GINE-$layers
        run_repeats ${dataset} GAT-$layers
        run_repeats ${dataset} GatedGCN-$layers
    done
}
function run_transformer_baselines {
    dataset=$1
    run_repeats ${dataset} GPS+None
    run_repeats ${dataset} GPS+BigBird
    run_repeats ${dataset} GPS+Performer
    run_repeats ${dataset} GPS+Transformer
    run_repeats ${dataset} Exphormer
}
function run_kmip {
    dataset=$1
    for lr in 0.0001 0.0005; do
    for layers in 4 8 12; do
    for kq_dim in 5 10; do
    run_repeats ${dataset} GPS+SparseAttention-${layers}l-${kq_dim} "optim.base_lr $lr name_tag lr-${lr}"
    done
    done
    done
}

function run_all {
    dataset=$1
    run_gnn_baselines ${dataset}
    run_transformer_baselines ${dataset}
    run_kmip ${dataset}
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
cfg_dir="configs/Large-experiment"
slurm_directive="
#SBATCH --partition=short
#SBATCH --time=12:00:00
#SBATCH --mem=60G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu_mem:16GB,gpu_sku:V100"
"


run_all ShapeNet-Part

run_all ModelNet10

run_all ModelNet40

run_all S3DIS

