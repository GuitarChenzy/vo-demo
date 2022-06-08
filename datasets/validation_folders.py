import torch.utils.data as data
import numpy as np
from imageio import imread
from path import Path
import os
import torch


def crawl_folders(folders_list, dataset='kitti'):
    img_ls, img_rs, depths_l, depths_r = [], [], [], []
    for folder in folders_list:
        folder_l, folder_r = folder + '02', folder + '03'
        img_l, img_r = sorted(folder_l.files('*.jpg')), sorted(folder_r.files('*.jpg'))
        if dataset == 'nyu':
            current_depth = sorted((folder / 'depth/').files('*.png'))
        elif dataset == 'kitti':
            depth_l, depth_r = sorted(folder_l.files('*.npy')), sorted(folder_r.files('*.npy'))
        img_ls.extend(img_l)
        img_rs.extend(img_r)
        depths_l.extend(depth_l)
        depths_r.extend(depth_r)
    # print(len(img_ls),len(depths_l))
    return img_ls, depths_l, img_ls, depths_r


class ValidationSet(data.Dataset):
    """A sequence data loader where the files are arranged in this way:
        root/scene_1/0000000.jpg
        root/scene_1/0000000.npy
        root/scene_1/0000001.jpg
        root/scene_1/0000001.npy
        ..
        root/scene_2/0000000.jpg
        root/scene_2/0000000.npy
        .

        transform functions must take in a list a images and a numpy array which can be None
    """

    def __init__(self, root, transform=None, dataset='kitti'):
        self.root = Path(root)
        scene_list_path = self.root / 'val.txt'
        self.scenes = [self.root / folder[:-3] for folder in open(scene_list_path)]
        self.transform = transform
        self.dataset = dataset
        self.img_ls, self.depth_ls, self.img_rs, self.depth_rs = crawl_folders(self.scenes, self.dataset)

    def __getitem__(self, index):
        img_l, img_r = imread(self.img_ls[index]).astype(np.float32), imread(self.img_rs[index]).astype(np.float32)

        if self.dataset == 'nyu':
            depth = torch.from_numpy(imread(self.depth[index]).astype(np.float32)).float() / 5000
        elif self.dataset == 'kitti':
            depth_l = torch.from_numpy(np.load(self.depth_ls[index]).astype(np.float32))
            depth_r = torch.from_numpy(np.load(self.depth_rs[index]).astype(np.float32))

        if self.transform is not None:
            img_l, _ = self.transform([img_l], None)
            img_r, _ = self.transform([img_r], None)

        return img_l[0], depth_l, img_r[0], depth_r

    def __len__(self):
        return len(self.img_ls)


if __name__ == '__main__':
    kitti = ValidationSet(root='/media/czy/DATA/Share/Kitti/kitti_256')
    print(kitti.__len__(), len(kitti.__getitem__(0)))
