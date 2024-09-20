method=$1
SEED=$2
dataset="ModelNet10"
name_tag="iterative"

cfg_file="configs/PC/${method}.yaml"
cfg_overrides="train.ckpt_best True name_tag ${name_tag} gnn.head graph dataset.format PyG-ModelNet10OnDisk metric_best accuracy wandb.project ${dataset} dataset.dir ${SCRATCH}/GraphGPS/datasets train.batch_size 8 optim.batch_accumulation 2 model.loss_fun cross_entropy"

out_dir="results/PC-iterative/${dataset}"
out_dir_prev_it="${out_dir}/${method}-${name_tag}/${SEED}"
main="$DATA/.conda/envs/exphormer/bin/python -O main.py --cfg ${cfg_file}"
common_params="out_dir ${out_dir} ${cfg_overrides}"

echo "Setting up environment ..."
module load Anaconda3
module load CUDA/12.0.0
source activate exphormer

echo "Making directory and redirecting output ..."
job_output_dir="$${out_dir}/${method}-${name_tag}/job_output/${SEED}"
mkdir -p $job_output_dir

timestamp=$(date +"%Y.%m-%d_%H:%M")

echo "Moving to scratch ..."
mkdir -p ${SCRATCH}/GraphGPS/datasets/${dataset}OnDisk
cp -r $DATA/GraphGPS/datasets/${dataset}OnDisk $SCRATCH/GraphGPS/datasets

echo "Looking for pretrained model at ${out_dir_prev_it}/ckpt/*.ckpt ..."
if compgen -G ${out_dir_prev_it}/ckpt/*.ckpt > /dev/null; then
    # find the latest checkpoint
    ckpt=$(ls ${out_dir_prev_it}/ckpt/*.ckpt | sort -n | tail -n 1)
    echo "Pretrained model found: ${ckpt}. Will continue training from this model"
    
    # extract the epoch number
    epoch=$(echo $ckpt | grep -o -E '[0-9]+\.ckpt' | tail -n 1)
    echo "starting from epoch: ${epoch}"

    common_params="${common_params} train.auto_resume True train.epoch_resume -1"
else
    echo "Pretrained model not found. Will train model from scratch"
fi

echo "Running program: ${main} --repeat 1 seed ${SEED} ${common_params} 2> >(tee ${job_output_dir}/${timestamp}.err >&2) | tee -a ${job_output_dir}/${timestamp}.out"
${main} --repeat 1 seed ${SEED} ${common_params} 2> >(tee ${job_output_dir}/${timestamp}.err >&2) | tee -a ${job_output_dir}/${timestamp}.out
