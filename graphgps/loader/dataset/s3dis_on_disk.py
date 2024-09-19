from collections import defaultdict
import os
import os.path as osp
import shutil
import time

import math
import torch
import random
from tqdm import tqdm

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    Dataset,
    extract_zip
)

import logging

from graphgps.transform.transforms import generate_knn_graph_from_pos


class S3DISOnDisk(Dataset):
    r"""The (pre-processed) Stanford Large-Scale 3D Indoor Spaces dataset from
    the `"3D Semantic Parsing of Large-Scale Indoor Spaces"
    <https://openaccess.thecvf.com/content_cvpr_2016/papers/Armeni_3D_Semantic_Parsing_CVPR_2016_paper.pdf>`_
    paper, containing point clouds of six large-scale indoor parts in three
    buildings with 12 semantic elements (and one clutter class).

    Args:
        root (str): Root directory where the dataset should be saved.
        test_area (int, optional): Which area to use for testing (1-6).
            (default: :obj:`6`)
        train (bool, optional): If :obj:`True`, loads the training dataset,
            otherwise the test dataset. (default: :obj:`True`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """

    url = ('https://shapenet.cs.stanford.edu/media/'
           'indoor3d_sem_seg_hdf5_data.zip')

    def __init__(self, root, test_area=6, train=True, transform=None,
                 pre_transform=None, pre_filter=None, max_num_chunks_in_memory = 2):
        assert test_area >= 1 and test_area <= 6
        self.test_area = test_area

        self.CHUNK_SIZE = 300        # hard coded for now: number of graphs to save in one file
        self.DEBUG_MAX_TTV_GRAPHS = None


        super().__init__(root, transform, pre_transform, pre_filter)

        # holds the chunks in memory, the oldest at index 0
        # if the number of chunks in memory exceeds max_num_chunks_in_memory, the oldest chunk is removed
        self.chunks_in_memory = []
        self.memory = {}
        self.max_num_chunks_in_memory = max_num_chunks_in_memory

        self.idx_split = torch.load(self.processed_paths[0])
        self.chunk_contents = torch.load(self.processed_paths[1])

        # for each graph id, store a tuple (chunk_id, idx_within_chunk)
        self.graph_id_to_chunk_location = {
            graph_id: (chunk_id, idx_within_chunk)
            for chunk_id, graph_ids in self.chunk_contents.items()
            for (idx_within_chunk, graph_id) in enumerate(graph_ids)
        }

        self.slices = None
        self.is_on_disk_dataset = True

        self.data = Data()

    @property
    def raw_file_names(self):
        return ['all_files.txt', 'room_filelist.txt']

    @property
    def processed_file_names(self):
        return ['s3dis_split_idxs.pt', 'chunk_contents.pt']

    def download(self):
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.unlink(path)
        shutil.rmtree(self.raw_dir)
        name = self.url.split('/')[-1].split('.')[0]
        os.rename(osp.join(self.root, name), self.raw_dir)

    def process(self):
        import h5py

        with open(self.raw_paths[0], 'r') as f:
            filenames = [x.split('/')[-1] for x in f.read().split('\n')[:-1]]

        with open(self.raw_paths[1], 'r') as f:
            rooms = f.read().split('\n')[:-1]

        # --- 1. decide which graphs go into which chunk ---

        # dict that contains the graph ids of each file, where we give an id to each graph
        file_contents = defaultdict(list)
        # a dict that contains the 'offset' of each file, i.e. the lowest graph id in that file
        file_offsets = {}
        # lists of train & test graph ids
        train_graph_ids = []
        test_graph_ids = []
        test_area = f'Area_{self.test_area}'

        logging.info("Scanning directory and assigning graphs to train/test sets...")
        largest_graph_id_so_far = -1
        for filename in tqdm(filenames):
            f = h5py.File(osp.join(self.raw_dir, filename))
            xs = torch.from_numpy(f['data'][:])
            ys = torch.from_numpy(f['label'][:]).to(torch.long)
            f.close()
            assert xs.size(0) == ys.size(0)

            ids_in_this_file = list(range(largest_graph_id_so_far+1, largest_graph_id_so_far+1+xs.size(0)))
            file_contents[filename] = (list(range(largest_graph_id_so_far+1, largest_graph_id_so_far+1+xs.size(0))))
            file_offsets[filename] = largest_graph_id_so_far+1
            largest_graph_id_so_far += xs.size(0)

            for i in ids_in_this_file:
                if test_area in rooms[i]:
                    test_graph_ids.append(i)
                else:
                    train_graph_ids.append(i)

        # we will regularly use this quantity for sanity checks
        total_num_graphs = len(train_graph_ids) + len(test_graph_ids)
        assert total_num_graphs == largest_graph_id_so_far+1
        assert total_num_graphs == sum([len(file_contents[filename]) for filename in filenames])

        # create list of val graph ids of the same size as the test graph ids
        random.shuffle(train_graph_ids)
        random.shuffle(test_graph_ids)
        val_graph_ids = train_graph_ids[:len(test_graph_ids)]
        train_graph_ids = train_graph_ids[len(test_graph_ids):]

        if self.DEBUG_MAX_TTV_GRAPHS is not None:
            train_graph_ids = train_graph_ids[:self.DEBUG_MAX_TTV_GRAPHS]
            test_graph_ids = test_graph_ids[:self.DEBUG_MAX_TTV_GRAPHS]
            val_graph_ids = val_graph_ids[:self.DEBUG_MAX_TTV_GRAPHS]
            total_num_graphs = len(train_graph_ids) + len(test_graph_ids) + len(val_graph_ids)

        print("num train graphs:", len(train_graph_ids))
        print("num val graphs:", len(val_graph_ids))
        print("num test graphs:", len(test_graph_ids))

        # a dict that maps a chunk number to a list of graph indices in that chunk
        # 'train{id}' chunks contain training graphs, 'test{id}' chunks contains test graphs, 'val{id}' chunks contain validation graphs
        self.chunk_contents = defaultdict(list)

        # allocate chunks
        num_train_chunks = math.ceil(len(train_graph_ids) / self.CHUNK_SIZE)
        num_test_chunks = math.ceil(len(test_graph_ids) / self.CHUNK_SIZE)
        num_val_chunks = math.ceil(len(val_graph_ids) / self.CHUNK_SIZE)
        for i in range(num_train_chunks):
            self.chunk_contents[f'train{i}'] = train_graph_ids[i*self.CHUNK_SIZE:(i+1)*self.CHUNK_SIZE]
        for i in range(num_test_chunks):
            self.chunk_contents[f'test{i}'] = test_graph_ids[i*self.CHUNK_SIZE:(i+1)*self.CHUNK_SIZE]
        for i in range(num_val_chunks):
            self.chunk_contents[f'val{i}'] = val_graph_ids[i*self.CHUNK_SIZE:(i+1)*self.CHUNK_SIZE]

        assert total_num_graphs == sum([len(graph_indices) for graph_indices in self.chunk_contents.values()])


        # --- 2. for each chunk, collect the graphs and save to disk ---

        for chunk_id, chunk_graph_ids in tqdm(list(self.chunk_contents.items())):
            # a dict {idx: graph} for all graphs in this chunk, where idx will determine the order
            chunk_graphs = {}

            for filename, file_graph_ids in file_contents.items():
                # collect the tuples (chunk_pos, file_pos) of all graphs that are in both this chunk and this file
                graphs_in_this_file = [
                    (
                        chunk_graph_ids.index(i),
                        i-file_offsets[filename]
                    )
                    for i in file_graph_ids
                    if i in chunk_graph_ids
                ]

                if len(graphs_in_this_file) == 0:
                    continue

                f = h5py.File(osp.join(self.raw_dir, filename))
                xs = torch.from_numpy(f['data'][:])
                ys = torch.from_numpy(f['label'][:]).to(torch.long)
                f.close()
                assert xs.size(0) == ys.size(0)

                for chunk_pos, file_pos in graphs_in_this_file:
                    # x: first 3 columns are positions, rest are other features (normals)
                    data = Data(x=xs[file_pos], y=ys[file_pos])

                    if self.pre_filter is not None and not self.pre_filter(data):
                        continue
                    # hardcoded extra pre-transform
                    # k=32 is chosen because it is the k for which PointViG (https://arxiv.org/pdf/2407.00921v1) performed best on S3DIS
                    data = generate_knn_graph_from_pos(data, k=32, distance_edge_attr=True, custom_pos=xs[file_pos, :, :3])
                    if self.pre_transform is not None:
                        data = self.pre_transform(data)

                    chunk_graphs[chunk_pos] = data

            # if everything went correctly, the chunk_graphs should now contain all graphs in this chunk, i.e. all indices from 0 to len(chunk_graphs)-1
            assert set(chunk_graphs.keys()) == set(range(len(self.chunk_contents[chunk_id])))

            # save the chunk to disk
            data_list = [chunk_graphs[i] for i in range(len(chunk_graphs))]
            torch.save(data_list, osp.join(self.processed_dir, f'{chunk_id}.pt'))
        
        # do sanity check and log the range of graph indices in each chunk
        logging.info(f"self.chunk_contents.keys(): {list(self.chunk_contents.keys())}")
        for chunk_name, graph_indices in self.chunk_contents.items():
            if len(graph_indices) == 0:
                logging.info(f"Chunk {chunk_name} is empty")
            elif len(graph_indices) == 1:
                logging.info(f"Chunk {chunk_name} only has 1 graph: should only happen if this is the last training graph and segmenting the dataset was a bit unlucky")
            else:
                logging.info(f"Chunk {chunk_name} contains graphs with indices {graph_indices}")

        self.train_mask = torch.tensor(train_graph_ids, dtype=torch.long)
        self.test_mask = torch.tensor(test_graph_ids, dtype=torch.long)
        self.val_mask = torch.tensor(val_graph_ids, dtype=torch.long)
        self.idx_split = {
            'train': self.train_mask,
            'val': self.val_mask,
            'test': self.test_mask,
        }

        torch.save(self.idx_split, self.processed_paths[0])
        torch.save(self.chunk_contents, self.processed_paths[1])

    def get_idx_split(self):
        if not hasattr(self, 'idx_split'):
            self.idx_split = torch.load(self.processed_paths[0])
        return self.idx_split
    
    def _get_chunk(self, chunk_id: str) -> list:
        if chunk_id in self.memory:
            return self.memory[chunk_id]
        else:
            # load from disk
            begin = time.time()
            chunk = torch.load(osp.join(self.processed_dir, f'{chunk_id}.pt'))
            logging.info(f"Loading chunk {chunk_id} from disk took {time.time()-begin:.2f} seconds")

            # if the number of chunks in memory exceeds max_num_chunks_in_memory, the oldest chunk is removed
            if len(self.chunks_in_memory) >= self.max_num_chunks_in_memory:
                oldest_chunk_id = self.chunks_in_memory.pop(0)
                del self.memory[oldest_chunk_id]

            self.chunks_in_memory.append(chunk_id)
            self.memory[chunk_id] = chunk
            return chunk
    
    def get(self, idx: int) -> Data:
        # get the chunk that contains the graph with index idx
        chunk_id, idx_within_chunk = self.graph_id_to_chunk_location[idx]

        # get the chunk
        chunk = self._get_chunk(chunk_id)

        # get the graph within the chunk
        return chunk[idx_within_chunk]

    def len(self) -> int:
        return len(self.graph_id_to_chunk_location)
    
    def get_num_classes(self) -> int:
        return 13   # 12 semantic elements + 1 clutter class