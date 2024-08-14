module load Anaconda3
module load CUDA/12.0.0
source activate exphormer

jupyter-notebook --no-browser --ip=0.0.0.0 --port 8888



# https://adam-streck.medium.com/creating-persistent-jupyter-notebooks-on-a-cluster-using-vs-code-slurm-and-conda-140b922a97a8