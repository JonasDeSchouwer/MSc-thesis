if [[ $1 == "" ]]; then
    partition="short"
else
    partition=$1
fi

if [[ $2 == "" ]]; then
    job_name="myJob"
else
    job_name=$2
fi

constraint=$3

if [[ $partition == "interactive" ]]; then
    time="4:00:00"
else
    time="11:00:00"
fi



echo "partition: $partition"
echo "job name: $job_name"
echo "time: $time"
echo "constraint: $constraint"

echo ""

srun --nodes=1 --partition=$partition --gres=gpu:1 --constraint=$constraint --mem=60G --clusters=htc --time=$time --pty --job-name=$job_name bash -i
