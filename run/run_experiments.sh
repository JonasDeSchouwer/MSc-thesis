#!/usr/bin/env bash

# Run this script from the project root dir.

environment_setup="
module load Anaconda3
module load CUDA/12.0.0
source activate LGI-env
"

function run_repeats {
    dataset=$1
    cfg_suffix=$2
    # The cmd line cfg overrides that will be passed to the main.py,
    # e.g. 'name_tag test01 gnn.layer_type gcnconv'
    cfg_overrides=$3

    cfg_file="${cfg_dir}/${dataset}-${cfg_suffix}.yaml"
    if [[ ! -f "$cfg_file" ]]; then
        echo "WARNING: Config does not exist: $cfg_file"
        echo "SKIPPING!"
        return 1
    fi

    main="$DATA/.conda/envs/graphgps/bin/python main.py --cfg ${cfg_file}"
    out_dir="results/${dataset}"  # <-- Set the output dir.
    common_params="out_dir ${out_dir} ${cfg_overrides}"

    echo "Run program: ${main}"
    echo "  output dir: ${out_dir}"

    time=`date +%m.%d-%H:%M`
    # Run each repeat as a separate job
    for SEED in {0..5}; do
        echo job name ${cfg_suffix}-${dataset}: seed ${SEED}
        echo ${main} --repeat 1 seed ${SEED} ${common_params}

        sbatch <<EOT
#!/bin/bash
        #SBATCH --output="output/${time}-%x-%j.out"
        #SBATCH --error="output/${time}-%x-%j.out"
        ${slurm_directive}
        #SBATCH --job-name=${cfg_suffix}-${dataset}
        ${environment_setup}
        ${main} --repeat 1 seed ${SEED} ${common_params}
EOT
    done
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
cfg_dir="configs/GPS"

DATASET="mnist"
slurm_directive="
#SBATCH --partition=medium
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
"

run_repeats ${DATASET} GCN "name_tag GCNwLapPE"
run_repeats ${DATASET} GPS-Transformer "name_tag GPSwLapPE.GINE+Transformer"
run_repeats ${DATASET} GPS-Performer "name_tag GPSwLapPE.GINE+Performer"
run_repeats ${DATASET} GPS-SparseAttention "name_tag GPSwLapPE.GINE+SparseMHA"


# DATASET="zinc"
# slurm_directive="--time=0-15:00:00 --mem=16G --gres=gpu:1 --cpus-per-task=4"
# run_repeats ${DATASET} GPS+RWSE "name_tag GPSwRWSE.10run"


# DATASET="cifar10"
# slurm_directive="--time=0-5:00:00 --mem=16G --gres=gpu:1 --cpus-per-task=4"
# run_repeats ${DATASET} GPS "name_tag GPSwLapPE.GatedGCN+Trf.10run"


# DATASET="pattern"
# slurm_directive="--time=0-6:00:00 --mem=16G --gres=gpu:1 --cpus-per-task=4"
# run_repeats ${DATASET} GPS "name_tag GPSwLapPE.GatedGCN+Trf.10run  wandb.project PATTERN-fix"


# DATASET="cluster"
# slurm_directive="--time=0-6:00:00 --mem=16G --gres=gpu:1 --cpus-per-task=4"
# run_repeats ${DATASET} GPS "name_tag GPSwLapPE.GatedGCN+Trf.10run  wandb.project CLUSTER-fix"


# DATASET="ogbg-molhiv"
# slurm_directive="--time=0-6:00:00 --mem=16G --gres=gpu:1 --cpus-per-task=4"
# run_repeats ${DATASET} GPS+RWSE "name_tag GPSwRWSE.GatedGCN+Trf.10run"


# DATASET="ogbg-molpcba"
# slurm_directive="--time=0-12:00:00 --mem=16G --gres=gpu:1 --cpus-per-task=4"
# run_repeats ${DATASET} GPS+RWSE "name_tag GPSwRWSE.10run"


# DATASET="ogbg-code2"
# slurm_directive="--time=1-6:00:00 --mem=24G --gres=gpu:1 --cpus-per-task=4"
# run_repeats ${DATASET} GPS "name_tag GPSnoPE.10run"


# DATASET="ogbg-ppa"  # NOTE: for ogbg-ppa we need SBATCH --mem=48G
# run_repeats ${DATASET} GPS "name_tag GPSnoPE.GatedGCN+Perf.lyr3.dim256.drp01.wd-5.10run"


# DATASET="pcqm4m"  # NOTE: for PCQM4Mv2 we need SBATCH --mem=48G and up to 3days runtime; run only one repeat
# slurm_directive="--time=3-00:00:00 --mem=48G --gres=gpu:1 --cpus-per-task=4"
# #run_repeats ${DATASET} GPS+RWSE "name_tag GPSwRWSE.small.lyr5.dim304  train.ckpt_best True"
# #run_repeats ${DATASET} GPSmedium+RWSE "name_tag GPSwRWSE.medium.lyr10.dim384.heads16.drp01.attndrp01.lr0002.e150  train.ckpt_best True"

# run_repeats ${DATASET} GPSmedium+RWSE "name_tag GPSwRWSE.medium train.ckpt_best True"
# run_repeats ${DATASET} GPSmedium+RWSE "name_tag GPSwRWSE.medium.gelu.linlr  optim.scheduler linear_with_warmup gnn.act gelu train.ckpt_best True"

# slurm_directive="--time=4-00:00:00 --mem=48G --gres=gpu:1 --cpus-per-task=4"
# run_repeats ${DATASET} GPSlarge+RWSE "name_tag GPSwRWSE.large train.ckpt_best True"


# DATASET="malnettiny"
# run_repeats ${DATASET} GPS-noPE  "name_tag GPS-noPE.GatedGCN+Perf.lyr5.dim64.10runs"
# run_repeats ${DATASET} GPS-noPE  "name_tag GPS-noPE.GatedGCN+Trf.lyr5.dim64.bs4.bacc4.10run  train.batch_size 4 optim.batch_accumulation 4 gt.layer_type CustomGatedGCN+Transformer"



################################################################################
##### SAN
################################################################################
# cfg_dir="configs/SAN"
# slurm_directive="--time=1-00:00:00 --mem=16G --gres=gpu:1 --cpus-per-task=4"

#DATASET="pattern"
#run_repeats ${DATASET} SAN "name_tag SAN.10run-fix  wandb.project PATTERN-fix"
#
#DATASET="cluster"
#run_repeats ${DATASET} SAN "name_tag SAN.10run-fix  wandb.project CLUSTER-fix"
