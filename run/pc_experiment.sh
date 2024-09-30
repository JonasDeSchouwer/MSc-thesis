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
    cfg_overrides="${cfg_overrides}"

    # --- Set the dataset-specific cfg overrides ---
    # if dataset is ShapeNet-Part
    if [[ $dataset == "ShapeNet-Part" ]]; then
        cfg_overrides="${cfg_overrides} gnn.head inductive_node dataset.format PyG-ShapeNet metric_best f1 wandb.project ShapeNet"
    # elif dataset is ModelNet10
    elif [[ $dataset == "ModelNet10" ]]; then
        cfg_overrides="${cfg_overrides} dataset.dir ${SCRATCH}/GraphGPS/datasets"
    # elif dataset is ModelNet40
    elif [[ $dataset == "ModelNet40" ]]; then
        cfg_overrides="${cfg_overrides} dataset.dir ${SCRATCH}/GraphGPS/datasets"
    # elif dataset is S3DIS
    elif [[ $dataset == "S3DIS" ]]; then
        cfg_overrides="${cfg_overrides} dataset.dir ${SCRATCH}/GraphGPS/datasets"
    fi

    cfg_file="configs/Large-experiment/${dataset}/${method}.yaml"
    if [[ ! -f "$cfg_file" ]]; then
        echo "WARNING: Config does not exist: $cfg_file"
        echo "SKIPPING!"
        return 1
    fi

    main="$DATA/.conda/envs/exphormer/bin/python -O main.py --cfg ${cfg_file}"
    out_dir="results/PC/${dataset}"  # <-- Set the output dir.
    common_params="out_dir ${out_dir} ${cfg_overrides}"

    echo ""
    echo "Run program: ${main}"
    echo "  output dir: ${out_dir}"
    
    if [[ $dataset == "ModelNet"* ]]; then
        move_to_scratch="mkdir -p ${SCRATCH}/GraphGPS/datasets/${dataset}OnDisk; cp -r $DATA/GraphGPS/datasets/${dataset}OnDisk $SCRATCH/GraphGPS/datasets"
        echo "Moving dataset to scratch"
    elif [[ $dataset == "S3DIS" ]]; then
        move_to_scratch="mkdir -p ${SCRATCH}/GraphGPS/datasets/S3DISOnDisk; cp $DATA/GraphGPS/datasets/S3DIS-temp/indoor3d_sem_seg_hdf5_data.zip $SCRATCH/GraphGPS/datasets/S3DISOnDisk"
        echo "Moving dataset zip file to scratch"
    else
        move_to_scratch=""
    fi

    time=`date +%m.%d-%H:%M`
    # Run each repeat as a separate job
    for SEED in {0..2}; do  # only 3 runs because my priority credits ran out
        echo job name ${method}-${dataset}: seed ${SEED}
        echo ${move_to_scratch}
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
${move_to_scratch}
${main} --repeat 1 seed ${SEED} ${common_params}
EOT
    done
}


function run_gnn_baselines {
    dataset=$1
    for layers in 5; do
        run_repeats ${dataset} GCN-$layers-100K
        run_repeats ${dataset} GINE-$layers-100K
        run_repeats ${dataset} GAT-$layers-100K
        run_repeats ${dataset} GatedGCN-$layers-100K
    done
}
function run_transformer_baselines {
    dataset=$1
    run_repeats ${dataset} GPS+None-100K
    run_repeats ${dataset} GPS+BigBird-100K
    run_repeats ${dataset} GPS+Performer-100K
    run_repeats ${dataset} GPS+Transformer-100K
    run_repeats ${dataset} Exphormer-100K
}
function run_kmip {
    dataset=$1
    for layers in 4; do
    for kq_dim in 5; do
    run_repeats ${dataset} GPS+SparseAttention-${layers}l-${kq_dim}-100K
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
########################### Experiments to run #################################
################################################################################



# --- S3DIS ---

slurm_directive="
#SBATCH --clusters=htc
#SBATCH --partition=medium
#SBATCH --time=1-23:00:00
#SBATCH --mem=60G
#SBATCH --gres='gpu:1'
#SBATCH --constraint='gpu_mem:32GB|gpu_mem:40GB|gpu_mem:48GB'
"

run_all S3DIS




# --- ModelNet10 ---

# BACK OF THE ENVELOPE MATH:
# BigBird-4l: 95h
# Performer-4l: 24h
# kMIP-4l-5kq: 33h
# Exphormer-4l: 24h
# GNNs: 13h

slurm_directive="
#SBATCH --clusters=htc
#SBATCH --partition=medium
#SBATCH --time=2-00:00:00
#SBATCH --mem=60G
#SBATCH --gres=gpu:1
#SBATCH --constraint='gpu_mem:40GB|gpu_mem:48GB'
"

run_repeats ModelNet10 GatedGCN-10-100K
run_gnn_baselines ModelNet10
run_repeats ModelNet10 GPS+Performer-100K
run_repeats ModelNet10 Exphormer-100K


slurm_directive="
#SBATCH --clusters=htc
#SBATCH --partition=long
#SBATCH --time=7-00:00:00
#SBATCH --mem=60G
#SBATCH --gres=gpu:1
#SBATCH --constraint='gpu_mem:40GB|gpu_mem:48GB'
"

run_repeats ModelNet10 GPS+BigBird-100K
run_kmip ModelNet10


# for testing

# slurm_directive="
# #SBATCH --clusters=htc
# #SBATCH --partition=devel
# #SBATCH --time=00:10:00
# #SBATCH --mem=60G
# #SBATCH --gres=gpu:1
# "

# run_repeats ModelNet10 GPS+Performer-100K
# run_repeats ModelNet10 Exphormer-100K
# run_repeats ModelNet10 GPS+BigBird-100K
# run_kmip ModelNet10
# run_gnn_baselines ModelNet10