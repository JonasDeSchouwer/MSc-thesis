if [[ $1 == "" ]]; then
    job_name="myJob"
else
    job_name=$1
fi

srun --nodes=1 --partition=interactive --gres=gpu:1 --mem=60G --clusters=htc --time=11:00:00 --pty --job-name=$job_name bash -i
