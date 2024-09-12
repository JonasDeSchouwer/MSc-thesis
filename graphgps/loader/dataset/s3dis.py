import os
import os.path as osp
import shutil

import torch

from torch_geometric.data import (Data, InMemoryDataset, download_url,
                                  extract_zip)


class S3DIS(InMemoryDataset):
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
                 pre_transform=None, pre_filter=None):
        assert test_area >= 1 and test_area <= 6
        self.test_area = test_area
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.idx_split = torch.load(self.processed_paths[1])
        self.slices['train_mask'] = self.idx_split['train']
        self.slices['val_mask'] = self.idx_split['val']
        self.slices['test_mask'] = self.idx_split['test']

    @property
    def raw_file_names(self):
        return ['all_files.txt', 'room_filelist.txt']

    @property
    def processed_file_names(self):
        return [f's3dis_processed.pt', f's3dis_split_idxs.pt']


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

        xs, ys = [], []
        for filename in filenames:
            f = h5py.File(osp.join(self.raw_dir, filename))
            xs += torch.from_numpy(f['data'][:]).unbind(0)
            ys += torch.from_numpy(f['label'][:]).to(torch.long).unbind(0)

        test_area = f'Area_{self.test_area}'
        data_list = []
        train_mask = []
        test_mask = []
        for i, (x, y) in enumerate(zip(xs, ys)):
            data = Data(pos=x[:, :3], x=x[:, 3:], y=y)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

            if test_area not in rooms[i]:
                train_mask.append(i)
            else:
                test_mask.append(i)

        self.train_mask = torch.tensor(train_mask, dtype=torch.long)
        self.test_mask = torch.tensor(test_mask, dtype=torch.long)

        # get a validation mask from the training set, of the same size as the test set
        self.train_mask = self.train_mask[torch.randperm(len(self.train_mask))]
        self.val_mask = self.train_mask[:len(self.test_mask)]
        self.train_mask = self.train_mask[len(self.test_mask):]
        self.idx_split = {
            'train': self.train_mask,
            'val': self.val_mask,
            'test': self.test_mask,
        }

        torch.save(self.collate(data_list), self.processed_paths[0])
        torch.save(self.idx_split, self.processed_paths[1])

    def get_idx_split(self):
        if not hasattr(self, 'idx_split'):
            self.idx_split = torch.load(self.processed_paths[1])
        return self.idx_split