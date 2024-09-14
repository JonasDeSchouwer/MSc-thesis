from collections import defaultdict
import glob
import logging
import os
import os.path as osp
import shutil
import time

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
from torch_geometric.io import read_off

from graphgps.transform.transforms import generate_knn_graph_from_pos, concat_x_and_pos


class ModelNetOnDisk(Dataset):
    r"""The ModelNet10/40 datasets from the `"3D ShapeNets: A Deep
    Representation for Volumetric Shapes"
    <https://people.csail.mit.edu/khosla/papers/cvpr2015_wu.pdf>`_ paper,
    containing CAD models of 10 and 40 categories, respectively.

    .. note::

        Data objects hold mesh faces instead of edge indices.
        To convert the mesh to a graph, use the
        :obj:`torch_geometric.transforms.FaceToEdge` as :obj:`pre_transform`.
        To convert the mesh to a point cloud, use the
        :obj:`torch_geometric.transforms.SamplePoints` as :obj:`transform` to
        sample a fixed number of points on the mesh faces according to their
        face area.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string, optional): The name of the dataset (:obj:`"10"` for
            ModelNet10, :obj:`"40"` for ModelNet40). (default: :obj:`"10"`)
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

    urls = {
        '10':
        'http://vision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip',
        '40': 'http://modelnet.cs.princeton.edu/ModelNet40.zip'
    }

    def __init__(self, root, name='10', train=True, transform=None,
                 pre_transform=None, pre_filter=None, max_num_chunks_in_memory=2):
        assert name in ['10', '40']
        self.name = name

        self.CHUNK_SIZE = 300
        self.DEBUG_MAX_2_TRAIN_CHUNKS = False
        self.DEBUG_MAX_1_TEST_CHUNK = False

        super().__init__(root, transform, pre_transform, pre_filter)

        # holds the chunks in memory, the oldest at index 0
        # if the number of chunks in memory exceeds max_num_chunks_in_memory, the oldest chunk is removed
        self.chunks_in_memory = []
        self.memory = {}
        self.max_num_chunks_in_memory = max_num_chunks_in_memory

        self.idx_split = torch.load(self.processed_paths[0])
        self.chunk_contents = torch.load(self.processed_paths[1])
        self.graph_id_to_chunk_id = { graph_id: chunk_id for chunk_id, graph_ids in self.chunk_contents.items() for graph_id in graph_ids }

        # assert that each value in self.chunk_contents is a contiguous range
        for graph_ids in self.chunk_contents.values():
            assert graph_ids == list(range(graph_ids[0], graph_ids[-1]+1))
        # compute the chunk offsets
        self.chunk_offsets = { chunk_id: graph_ids[0] for chunk_id, graph_ids in self.chunk_contents.items() }

        self.slices = None
        self.is_on_disk_dataset = True

        self.data = Data()

    @property
    def raw_file_names(self):
        return [
            'bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor',
            'night_stand', 'sofa', 'table', 'toilet'
        ]

    @property
    def processed_file_names(self):
        return ['modelnet_split_idxs.pt', 'chunk_contents.pt']

    def download(self):
        path = download_url(self.urls[self.name], self.root)
        extract_zip(path, self.root)
        os.unlink(path)
        folder = osp.join(self.root, f'ModelNet{self.name}')
        shutil.rmtree(self.raw_dir)
        os.rename(folder, self.raw_dir)

        # Delete osx metadata generated during compression of ModelNet10
        metadata_folder = osp.join(self.root, '__MACOSX')
        if osp.exists(metadata_folder):
            shutil.rmtree(metadata_folder)

    def process(self):
        categories = glob.glob(osp.join(self.raw_dir, '*', ''))
        categories = sorted([x.split(os.sep)[-2] for x in categories])

        # collect train and test graphs in a list, before saving them to disk
        train_graphs_buffer = []
        test_graphs_buffer = []
        
        # a dict that maps a chunk number to a list of graph indices in that chunk
        self.chunk_contents = defaultdict(list)

        train_paths = []
        test_paths = []
        for split in ('train', 'test'):
            for target, category in enumerate(categories):
                folder = osp.join(self.raw_dir, category, split)
                if split == 'train':
                    train_paths.extend(glob.glob(f'{folder}/{category}_*.off'))
                else:
                    test_paths.extend(glob.glob(f'{folder}/{category}_*.off'))

        # shuffle the train and test paths
        random.shuffle(train_paths)
        random.shuffle(test_paths)

        # save the chunks of the train set to disk
        current_train_chunk_id = 0
        for path in train_paths:
            data: Data = read_off(path)
            data.y = torch.tensor([target])
            train_graphs_buffer.append(data)

            data = self._filter_and_pre_transform(data)

            # if the buffer is full or the last graph in the train set is reached, save the buffer to disk
            if len(train_graphs_buffer) >= self.CHUNK_SIZE or path == train_paths[-1]:
                last_saved_train_graph_id = -1 \
                    if current_train_chunk_id == 0 \
                    else self.chunk_contents[f'train{current_train_chunk_id-1}'][-1]
                start = last_saved_train_graph_id+1
                end = last_saved_train_graph_id+len(train_graphs_buffer)
                logging.info(f"Saving graphs {start} to {end} to Chunk train{current_train_chunk_id}")
                torch.save(train_graphs_buffer, osp.join(self.processed_dir, f'train{current_train_chunk_id}.pt'))
                self.chunk_contents[f'train{current_train_chunk_id}'] = list(range(start, end+1))
                current_train_chunk_id += 1
                train_graphs_buffer = []

            if self.DEBUG_MAX_2_TRAIN_CHUNKS and current_train_chunk_id == 2:
                break

        # save the chunks of the test set to disk
        current_test_chunk_id = 0
        for path in test_paths:
            data: Data = read_off(path)
            data.y = torch.tensor([target])
            test_graphs_buffer.append(data)

            data = self._filter_and_pre_transform(data)

            # if the buffer is full or the last graph in the test set is reached, save the buffer to disk
            if len(test_graphs_buffer) >= self.CHUNK_SIZE or path == test_paths[-1]:
                last_saved_test_graph_id = -1 \
                    if current_test_chunk_id == 0 \
                    else self.chunk_contents[f'test{current_test_chunk_id-1}'][-1]
                start = last_saved_test_graph_id+1
                end = last_saved_test_graph_id+len(test_graphs_buffer)
                logging.info(f"Saving graphs {start} to {end} to Chunk test{current_test_chunk_id}")
                torch.save(test_graphs_buffer, osp.join(self.processed_dir, f'test{current_test_chunk_id}.pt'))
                self.chunk_contents[f'test{current_test_chunk_id}'] = list(range(start, end+1))
                current_test_chunk_id += 1
                test_graphs_buffer = []
            
            if self.DEBUG_MAX_1_TEST_CHUNK and current_test_chunk_id == 1:
                break

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        # no shuffling of the train and test masks, because the order of the graphs in the chunks is already random
        chunkwise_train_mask = [v for k, v in self.chunk_contents.items() if 'train' in k]
        chunkwise_test_mask = [v for k, v in self.chunk_contents.items() if 'test' in k]
        train_mask = sum(chunkwise_train_mask, start=[])
        test_mask = sum(chunkwise_test_mask, start=[])
        self.train_mask = torch.tensor(train_mask, dtype=torch.long)
        self.test_mask = torch.tensor(test_mask, dtype=torch.long)

        print("test_mask:", len(self.test_mask))
        self.val_mask = self.train_mask[:len(self.test_mask)]
        self.train_mask = self.train_mask[len(self.test_mask):]
        self.idx_split = {
            'train': self.train_mask,
            'val': self.val_mask,
            'test': self.test_mask,
        }
        print("train_mask", len(self.train_mask))
        print("val_mask", len(self.val_mask))

        torch.save(self.idx_split, self.processed_paths[0])
        torch.save(self.chunk_contents, self.processed_paths[1])

    def _filter_and_pre_transform(self, data: Data):
        if self.pre_filter is not None and not self.pre_filter(data):
            return None
        
        # hardcoded extra pre-transforms
        # k=8 is chosen because it is the k for which PointViG (https://arxiv.org/pdf/2407.00921v1) performed best on ModelNet40
        # TODO: check if this is the correct syntax
        data = concat_x_and_pos(data)

        data = generate_knn_graph_from_pos(data, k=8, distance_edge_attr=True)
        if self.pre_transform is not None:
            data = self.pre_transform(data)

        if self.pre_transform is not None:
            data = self.pre_transform(data)
        
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}{self.name}({len(self)})'
    
    def get_idx_split(self):
        if not hasattr(self, 'idx_split'):
            self.idx_split = torch.load(self.processed_paths[1])
        if self.idx_split is None:
            raise ValueError("self.idx_split is None. This should not happen.")
        return self.idx_split
    
    def _get_idx_within_chunk(self, graph_id: int, chunk_id: str) -> int:
        return graph_id - self.chunk_offsets[chunk_id]
    
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
        chunk_id = self.graph_id_to_chunk_id[idx]

        # get the chunk
        chunk = self._get_chunk(chunk_id)

        # get the graph within the chunk
        idx_within_chunk = self._get_idx_within_chunk(idx, chunk_id)
        return chunk[idx_within_chunk]

    def len(self) -> int:
        return len(self.graph_id_to_chunk_id)
    
    def get_num_classes(self) -> int:
        if self.name == '10':
            return 10
        elif self.name == '40':
            return 40
        else:
            raise ValueError(f"Unexpected name '{self.name}'")