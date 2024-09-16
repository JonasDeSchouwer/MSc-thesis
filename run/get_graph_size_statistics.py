import torch

import os
import os.path as osp
import numpy as np

import time
import matplotlib.pyplot as plt
import argparse
from torch_geometric.data import Data


def get_graph_size_statistics_from_chunks(processed_dir):
    # get all chunks: we assume the chunks are the files in processed_dir that start with 'train' or 'test'
    chunk_names = [fname for fname in os.listdir(processed_dir) if fname.startswith('train') or fname.startswith('test')]

    node_nums = []
    edge_nums = []
    is_directed = False

    n_graphs_processed = 0

    # loop over all chunks
    for chunk_name in chunk_names:
        chunk_path = osp.join(processed_dir, chunk_name)
        begin = time.time()
        chunk = torch.load(chunk_path)
        end = time.time()
        print(f"Loading {chunk_name} took {end - begin:.2f} seconds")
        # chunk is a list of Data objects

        for graph in chunk:
            graph: Data
            node_nums.append(graph.x.size(0))
            edge_nums.append(graph.edge_index.size(1))

            # estimate whether the dataset has directed graphs based on the first 20 graphs
            if n_graphs_processed < 20:
                if graph.is_directed():
                    is_directed = True

            n_graphs_processed += 1

    return node_nums, edge_nums, is_directed


def get_graph_size_statistics(processed_dir, force_recompute=False, plot=True):
    # if statistics have been previously computed, load them and proceed. Otherwise, compute them
    if osp.exists(osp.join(processed_dir, "graph_size_statistics.pt")) and not force_recompute:
        node_nums, edge_nums, is_directed = torch.load(osp.join(processed_dir, "graph_size_statistics.pt"))
        print(f"Loaded precomputed statistics from {osp.join(processed_dir, 'graph_size_statistics.pt')}")
    else:
        node_nums, edge_nums, is_directed = get_graph_size_statistics_from_chunks(processed_dir)

    # get number of graphs
    n_graphs = len(node_nums)
    print(f"Number of graphs: {n_graphs}")

    # get whether the dataset has directed graphs
    print(f"Estimated to be directed: {is_directed}")
    
    # get average node count
    node_avg = sum(node_nums) / len(node_nums)
    print(f"Average node count: {node_avg}")

    # get average edge count
    edge_avg = sum(edge_nums) / len(edge_nums)
    print(f"Average edge count: {edge_avg}")

    # get largest graph in terms of #nodes
    largest_node_count = max(node_nums)
    print(f"Largest node count: {largest_node_count}")

    # get largest graph in terms of #edges
    largest_edge_count = max(edge_nums)
    print(f"Largest edge count: {largest_edge_count}")

    # display a histogram of node counts using plt
    if plot:
        plt.figure()
        logbins = np.logspace(np.log10(min(node_nums)),np.log10(max(node_nums)),num=50)
        plt.hist(node_nums, bins=logbins)
        plt.xscale('log')
        plt.savefig(osp.join(processed_dir, "node_histogram.png"))
        plt.show()

    # save (node_nums, edge_nums, is_directed) to a file and load them from that file
    torch.save((node_nums, edge_nums, is_directed), osp.join(processed_dir, "graph_size_statistics.pt"))

    return node_nums, edge_nums, is_directed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get graph size statistics')
    parser.add_argument('dir', type=str, help='Directory containing the processed data')
    parser.add_argument('--force_recompute', action='store_true', help='Force recomputation of statistics')
    parser.add_argument('--no_plot', dest='plot', action='store_false', help='Do not plot the histogram')
    args = parser.parse_args()
    get_graph_size_statistics(args.dir, args.force_recompute, args.plot)

