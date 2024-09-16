import torch

import os
import os.path as osp

import time
import matplotlib.pyplot as plt
import argparse
import math
from torch_geometric.data import Data
import sys

# add the parent directory to the path to import the transforms module
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
from graphgps.transform.transforms import generate_knn_graph_from_pos

def process_all_chunks(processed_dir, max_edges=None):
    # get all chunks: we assume the chunks are the files in processed_dir that start with 'train' or 'test'
    chunk_names = [fname for fname in os.listdir(processed_dir) if fname.startswith('train') or fname.startswith('test')]

    n_graphs_processed = 0
    n_graphs_modified = 0

    # loop over all chunks
    for chunk_name in chunk_names:
        chunk_path = osp.join(processed_dir, chunk_name)
        begin = time.time()
        chunk = torch.load(chunk_path)
        end = time.time()
        print(f"Loading {chunk_name} took {end - begin:.2f} seconds")
        # chunk is a list of Data objects

        for i in range(len(chunk)):
            graph: Data = chunk[i]
            n_nodes = graph.x.size(0)
            n_edges = graph.edge_index.size(1)
            if n_edges > max_edges:
                print(f"Graph {n_graphs_processed} has {n_edges} edges, which is more than the maximum allowed of {max_edges}")
                k = math.floor(max_edges / n_nodes)
                print(f"Generating a kNN graph with k={k} => {k * n_nodes} edges")

                if hasattr(graph, 'edge_attr') and graph.edge_attr is not None:
                    # if there used to be an edge_attr, generate the distances as a new edge_attr
                    graph.edge_attr = None
                    generate_knn_graph_from_pos(graph, k, distance_edge_attr=True)
                else:
                    generate_knn_graph_from_pos(graph, k, distance_edge_attr=False)
                
                n_graphs_modified += 1
            
            n_graphs_processed += 1

        # update the chunk
        torch.save(chunk, chunk_path)

    print(f"Modified {n_graphs_modified} graphs out of {n_graphs_processed} processed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Recompute the k-NN graph for all graphs in a processed dataset that are too large')
    parser.add_argument('dir', type=str, help='Directory containing the processed data')
    parser.add_argument('max_edges', type=int, help='Maximum number of edges allowed in a graph')
    args = parser.parse_args()
    process_all_chunks(args.dir, args.max_edges)

