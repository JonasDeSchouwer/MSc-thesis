# Bringing k-MIP Attention to Graph Transformers


In this work, we introduce the k-MIP Graph Transformer, which is based on the k-Maximum Inner Product (k-MIP) attention mechanism and the [GraphGPS](https://github.com/rampasek/GraphGPS) framework. The complete report can be found [here](https://github.com/JonasDeSchouwer/MSc-thesis/blob/main/Thesis.pdf).

The k-MIP Graph Transformer is:

- **Efficient:** The k-MIP self-attention mechanism is two orders of magnitude faster than full attention and has a negligible memory footprint, allowing us to scale to graphs with up to 500K nodes.
- **Versatile:** The k-MIP-GT incorporates edge features and supports node-level, graph-level, and edge-level tasks.
- **Performant:** We have demonstrated results competitive with prominent graph Transformers across a variety of graph learning tasks, with graphs ranging from 20 to 500K nodes.
- **Expressive:** We have established universal approximation guarantees for the k-MIP Graph Transformer, analogous to those previously established for full-attention Transformers and graph Transformers.

This repository was used to run the experiments in the following sections:

- 6.1: Performance on Various Small-Graph Datasets
- 6.4: Influence of $k$
- 6.5: Influence of $d_{kq}$
- 6.6: Scaling to Datasets with Larger Graphs
- 6.8: Inspecting the Attention Graphs

The experiments in the sections 6.2, 6.3, and 6.7 were run with our other repository: Efficient-k-MIP-Attention.


### Environment setup with conda

```bash
conda create -n kmipgt python=3.9
conda activate kmipgt

conda install pytorch=1.10 torchvision torchaudio -c pytorch -c nvidia
conda install pyg=2.0.4 -c pyg -c conda-forge

# RDKit is required for OGB-LSC PCQM4Mv2 and datasets derived from it.
# h5py is required for S3DIS  
pip install fsspec rdkit torchmetrics performer-pytorch ogb tensorboardX wandb pykeops ipykernel h5py cupy-cuda12x

conda clean --all
```

### Navigating the codebase

We highlight some important files and folders in the codebase.

| File/Folder                          | Description                                                                 |
|----------------------------|-----------------------------------------------------------------------------|
| `configs`                            | Contains configuration files for different experiments and datasets.        |
| `run`                                | Directory for scripts to execute various tasks and experiments. Importantly, this directory contains `small_experiment.sh`, `large_experiment.sh` and `pc_experiment.sh` that were used to batch experiments.             |
| `graphgps/layer/sparse_attention_layer.py` | Implementation of the sparse attention layer used in the k-MIP Graph Transformer. |
| `plotting` | Directory that contains the notebooks used for plotting. Warning: they are messy. |


### Training the k-MIP Graph Transformer

Training the k-MIP Graph Transformer proceeds as follows.

```bash
conda activate kmipgt

# Running k-MIP-GT for Cifar10
python main.py --cfg configs/Exphormer/cifar10/GPS+SparseAttention.yaml wandb.use False

# Running k-MIP-GT for ShapeNet
python main.py --cfg configs/Large-experiment/ShapeNet-Part/GPS+SparseAttention-8l.yaml wandb.use False
```
You may override any config key by adding it as extra argument, like we have done above for `wandb.use`



<!-- ## Citation

Our work can be cited using the following bibtex:
```bibtex

``` -->
