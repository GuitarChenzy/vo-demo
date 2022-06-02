import torch.utils.data as data
import numpy as np
from imageio import imread
from path import Path
import random
import os


def load_as_float(path):
    return imread(path).astype(np.float32)


class SequenceFolder(data.Dataset):
    """A sequence data loader where the files are arranged in this way:
        root/scene_1/0000000.jpg
        root/scene_1/0000001.jpg
        ..
        root/scene_1/cam.txt
        root/scene_2/0000000.jpg
        .
        transform functions must take in a list a images and a numpy array (usually intrinsics matrix)
    """

    def __init__(self, root, seed=None, train=True, sequence_length=3, transform=None, skip_frames=1, dataset='kitti'):
        np.random.seed(seed)
        random.seed(seed)
        self.root = Path(root)
        scene_list_path = self.root / 'train.txt' if train else self.root / 'val.txt'
        self.scenes = set([self.root / folder[:-3] for folder in open(scene_list_path)])
        self.transform = transform
        self.dataset = dataset
        self.k = skip_frames
        # self.crawl_folders(sequence_length)
        self.crwal_lr_folders()

    def crawl_folders(self, sequence_length):
        # k skip frames
        sequence_set = []
        demi_length = (sequence_length - 1) // 2
        shifts = list(range(-demi_length * self.k, demi_length * self.k + 1, self.k))
        shifts.pop(demi_length)
        for scene in self.scenes:
            intrinsics = np.genfromtxt(scene / 'cam.txt').astype(np.float32).reshape((3, 3))
            imgs = sorted(scene.files('*.jpg'))

            if len(imgs) < sequence_length:
                continue
            for i in range(demi_length * self.k, len(imgs) - demi_length * self.k):
                sample = {'intrinsics': intrinsics, 'tgt': imgs[i], 'ref_imgs': []}
                for j in shifts:
                    sample['ref_imgs'].append(imgs[i + j])
                sequence_set.append(sample)
        random.shuffle(sequence_set)
        self.samples = sequence_set

    # 构造双目图像对和对应的相机内参
    def crwal_lr_folders(self):
        # print(self.scenes)
        sequence_set = []
        for scene in self.scenes:
            scene_l, scene_r = scene + '02', scene + '03'
            intrinsics_l = np.genfromtxt(scene_l + '/cam.txt').astype(np.float32).reshape((3, 3))
            intrinsics_r = np.genfromtxt(scene_r + '/cam.txt').astype(np.float32).reshape((3, 3))
            img_l, img_r = sorted(scene_l.files('*.jpg')), sorted(scene_r.files('*.jpg'))
            # print(scene, img_l[0], img_r[0], img_l[1])
            for i in range(len(img_l)):
                sample = {'intrinsics_2': intrinsics_l, 'intrinsics_3': intrinsics_r,
                          'img_l': img_l[i], 'img_r': img_r[i]}
                sequence_set.append(sample)
        random.shuffle(sequence_set)
        self.samples = sequence_set

    def __getitem__(self, index):
        sample = self.samples[index]
        img_l, img_r = load_as_float(sample['img_l']), load_as_float(sample['img_r'])
        if self.transform is not None:
            img_l, intrinsics_l = self.transform(img_l, np.copy(sample['intrinsics_l']))
            img_r, intrinsics_r = self.transform(img_r, np.copy(sample['intrinsics_r']))
        else:
            intrinsics_l, intrinsics_r = np.copy(sample['intrinsics_l']), np.copy(sample['intrinsics_r'])
        return img_l, img_r, intrinsics_l, intrinsics_r

    def __len__(self):
        return len(self.samples)


if __name__ == '__main__':
    kitti = SequenceFolder(root='/media/czy/DATA/Share/Kitti/kitti_256')
    # len = 21288
    print(kitti.__len__(), kitti.samples[0])
