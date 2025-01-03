import logging

import torch
import torch_geometric
from torch_geometric.utils import subgraph
from tqdm import tqdm
import os.path as osp
import time


def pre_transform_in_memory(dataset, transform_func, show_progress=False):
    """Pre-transform already loaded PyG dataset object.

    Apply transform function to a loaded PyG dataset object so that
    the transformed result is persistent for the lifespan of the object.
    This means the result is not saved to disk, as what PyG's `pre_transform`
    would do, but also the transform is applied only once and not at each
    data access as what PyG's `transform` hook does.

    Implementation is based on torch_geometric.data.in_memory_dataset.copy

    Args:
        dataset: PyG dataset object to modify
        transform_func: transformation function to apply to each data example
        show_progress: show tqdm progress bar
    """
    if transform_func is None:
        return dataset

    data_list = [transform_func(dataset.get(i))
                 for i in tqdm(range(len(dataset)),
                               disable=not show_progress,
                               mininterval=10,
                               miniters=len(dataset)//20)]
    data_list = list(filter(None, data_list))

    dataset._indices = None
    dataset._data_list = data_list
    dataset.data, dataset.slices = dataset.collate(data_list)


def pre_transform_on_disk(dataset, transform_func, transform_name, show_progress=False):
    """Pre-transform already loaded PyG dataset object.

    Apply transform function to a loaded PyG dataset object so that
    the transformed result is persistent for the lifespan of the object.
    The result is saved to disk, like what PyG's `pre_transform` would do.
    """
    if transform_func is None:
        return dataset

    for chunk_id in tqdm(list(dataset.chunk_contents.keys()),
                        disable=not show_progress):
        chunk_path = osp.join(dataset.processed_dir, f'{chunk_id}.pt')
        transformed_chunk_path = osp.join(dataset.processed_dir, f'{chunk_id}_transformed_{transform_name}.pt')

        logging.info(transformed_chunk_path)

        if osp.exists(transformed_chunk_path):
            continue

        chunk = torch.load(chunk_path)

        transformed_chunk = []
        for graph in chunk:
            transformed_chunk.append(transform_func(graph))
        
        torch.save(transformed_chunk, transformed_chunk_path)


def generate_splits(data, g_split):
  n_nodes = len(data.x)
  train_mask = torch.zeros(n_nodes, dtype=bool)
  valid_mask = torch.zeros(n_nodes, dtype=bool)
  test_mask = torch.zeros(n_nodes, dtype=bool)
  idx = torch.randperm(n_nodes)
  val_num = test_num = int(n_nodes * (1 - g_split) / 2)
  train_mask[idx[val_num + test_num:]] = True
  valid_mask[idx[:val_num]] = True
  test_mask[idx[val_num:val_num + test_num]] = True
  data.train_mask = train_mask
  data.val_mask = valid_mask
  data.test_mask = test_mask
  return data


def typecast_x(data, type_str):
    if type_str == 'float':
        data.x = data.x.float()
    elif type_str == 'long':
        data.x = data.x.long()
    else:
        raise ValueError(f"Unexpected type '{type_str}'.")
    return data


def concat_x_and_pos(data):
    if data.x is None:
        data.x = data.pos
    else:
        data.x = torch.cat((data.x, data.pos), 1)
    return data

def move_node_feat_to_x(data):
    """For ogbn-proteins, move the attribute node_species to attribute x."""
    data.x = data.node_species
    return data

def clip_graphs_to_size(data, size_limit=5000):
    if hasattr(data, 'num_nodes'):
        N = data.num_nodes  # Explicitly given number of nodes, e.g. ogbg-ppa
    else:
        N = data.x.shape[0]  # Number of nodes, including disconnected nodes.
    if N <= size_limit:
        return data
    else:
        logging.info(f'  ...clip to {size_limit} a graph of size: {N}')
        if hasattr(data, 'edge_attr'):
            edge_attr = data.edge_attr
        else:
            edge_attr = None
        edge_index, edge_attr = subgraph(list(range(size_limit)),
                                         data.edge_index, edge_attr)
        if hasattr(data, 'x'):
            data.x = data.x[:size_limit]
            data.num_nodes = size_limit
        else:
            data.num_nodes = size_limit
        if hasattr(data, 'node_is_attributed'):  # for ogbg-code2 dataset
            data.node_is_attributed = data.node_is_attributed[:size_limit]
            data.node_dfs_order = data.node_dfs_order[:size_limit]
            data.node_depth = data.node_depth[:size_limit]
        data.edge_index = edge_index
        if hasattr(data, 'edge_attr'):
            data.edge_attr = edge_attr
        return data


def generate_knn_graph_from_pos(data: torch_geometric.data.Data, k: int, distance_edge_attr: bool = True, custom_pos=None):
    if custom_pos is not None:
        pos = custom_pos
    else:
        assert hasattr(data, 'pos'), 'Data object does not have a pos attribute'
        pos = data.pos
    # compute on cuda if available
    if torch.cuda.is_available():
        pos = pos.cuda()
    edge_index = torch_geometric.nn.knn_graph(
        pos,
        k=k,
        loop=False,
    )
    data.edge_index = edge_index.to(data.x.device)
    
    # compute the distance as edge attribute
    if distance_edge_attr:
        edge_attr = torch.norm(pos[edge_index[0]] - pos[edge_index[1]], dim=1).unsqueeze(-1).to(data.x.device)
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            data.edge_attr = torch.cat([data.edge_attr, edge_attr], dim=1)
        else:
            data.edge_attr = edge_attr
    else:
        if not hasattr(data, 'edge_attr'):
            data.edge_attr = None

    return data


def generate_chain_graph(data: torch_geometric.data.Data):
    n_nodes = data.x.size(0)
    edge_index = torch.stack([torch.arange(n_nodes - 1), torch.arange(1, n_nodes)], dim=0)
    data.edge_index = edge_index
    return data