from collections import defaultdict
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
        self.chunk_offsets = { chunk_id: graph_ids[0] for chunk_id, graph_ids in self.chunk_contents.items() }

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

        test_area = f'Area_{self.test_area}'

        # collect train and test graphs in a list, before saving them to disk
        train_graphs_buffer = []
        test_graphs_buffer = []

        # a dict that maps a chunk number to a list of graph indices in that chunk
        # 'test' is a special chunk that contains all the test graphs
        self.chunk_contents = defaultdict(list)

        # current chunk id to fill
        current_train_chunk_id = 0

        # only used to check if this is a test or a train graph
        num_graphs_processed = 0
        
        for filename in tqdm(filenames):
            f = h5py.File(osp.join(self.raw_dir, filename))
            xs = torch.from_numpy(f['data'][:])
            ys = torch.from_numpy(f['label'][:]).to(torch.long)
            f.close()
            assert xs.size(0) == ys.size(0)

            for i in range(xs.size(0)):
                # x: first 3 columns are positions, rest are other features (normals)
                data = Data(x=xs[i], y=ys[i])

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                # hardcoded extra pre-transform
                # k=32 is chosen because it is the k for which PointViG (https://arxiv.org/pdf/2407.00921v1) performed best on S3DIS
                data = generate_knn_graph_from_pos(data, k=32, distance_edge_attr=True, custom_pos=xs[i, :, :3])
                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                if test_area in rooms[num_graphs_processed]:
                    test_graphs_buffer.append(data)
                else:
                    train_graphs_buffer.append(data)

                num_graphs_processed += 1

                # if the buffer is full, save the graphs to disk
                if len(train_graphs_buffer) >= self.CHUNK_SIZE:
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

            # if num_graphs_processed >= self.DEBUG_MAX_NUM_GRAPHS:
            #     break

        # save remaining train graphs to disk
        last_saved_train_graph_id = -1 \
            if current_train_chunk_id == 0 \
            else self.chunk_contents[f'train{current_train_chunk_id-1}'][-1]
        start = last_saved_train_graph_id+1
        end = last_saved_train_graph_id+len(train_graphs_buffer)
        logging.info(f"Saving graphs {start} to {end} to Chunk train{current_train_chunk_id}")
        torch.save(train_graphs_buffer, osp.join(self.processed_dir, f'train{current_train_chunk_id}.pt'))
        self.chunk_contents[f'train{current_train_chunk_id}'] = list(range(start, end+1))
        current_train_chunk_id += 1
        del train_graphs_buffer

        # save the test chunk to disk
        logging.info(f"Saving {len(test_graphs_buffer)} test graphs to Chunk test")
        torch.save(test_graphs_buffer, osp.join(self.processed_dir, 'test.pt'))
        self.chunk_contents['test'] = list(range(num_graphs_processed-len(test_graphs_buffer), num_graphs_processed))
        del test_graphs_buffer
        
        # do sanity check and log the range of graph indices in each chunk
        logging.info("self.chunk_contents.keys():", list(self.chunk_contents.keys()))
        for chunk_name, graph_indices in self.chunk_contents.items():
            if len(graph_indices) == 0:
                logging.info(f"Chunk {chunk_name} is empty")
            elif len(graph_indices) == 1:
                logging.info(f"Chunk {chunk_name} only has 1 graph: should only happen if this is the last training graph and segmenting the dataset was a bit unlucky")
            else:
                logging.info(f"Chunk {chunk_name} contains graphs with indices {graph_indices[0]} to {graph_indices[-1]}")


        # shuffle the train_mask chunk wise, to retain fast data access but also shuffle the data
        chunkwise_train_mask = [v for k, v in self.chunk_contents.items() if 'train' in k]
        random.shuffle(chunkwise_train_mask)
        
        train_mask = sum(chunkwise_train_mask, start=[])
        self.train_mask = torch.tensor(train_mask, dtype=torch.long)
        test_mask = self.chunk_contents['test']
        self.test_mask = torch.tensor(test_mask, dtype=torch.long)

        # get a validation mask from the training set, of the same size as the test set
        print("test_mask:", len(self.test_mask))
        self.val_mask = self.train_mask[:len(self.test_mask)]
        self.train_mask = self.train_mask[len(self.test_mask):]
        self.idx_split = {
            'train': self.train_mask,
            'val': self.val_mask,
            'test': self.test_mask,
        }
        print("train_mask:", len(self.train_mask))
        print("val_mask:", len(self.val_mask))

        torch.save(self.idx_split, self.processed_paths[0])
        torch.save(self.chunk_contents, self.processed_paths[1])

    def get_idx_split(self):
        if not hasattr(self, 'idx_split'):
            self.idx_split = torch.load(self.processed_paths[0])
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
        return 13   # 12 semantic elements + 1 clutter class