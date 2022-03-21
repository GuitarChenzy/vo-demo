import imp
from PIL import Image, ImageFile
import numpy as np
from glob import glob
import re
from torch.utils.data import Dataset, DataLoader

ImageFile.LOAD_TRUNCATED_IMAGES = True


class KITTIDataSet(Dataset):
    """KITTI VO dataset"""

    def __init__(self, dir_data, dir_label, samples='i0', phase=None, seq=None):
        super().__init__()
        self.dir_data = dir_data
        self.dir_label = dir_label
        self.samples = samples
        self.phase = phase
        self.seq = seq

        if self.phase == 'Test':
            self.l1, self.l2, self.label = self.load_data_test()
        else:
            self.l1, self.l2, self.label = self.load_data()

        assert (len(self.l1) == len(self.l2)) and (
                len(self.l1) == len(self.label)), 'Length must be equal!'

    def load_data_test(self):
        img_list = glob(
            self.dir_data + '/{:02d}/image_3/*.png'.format(self.seq))
        img_list.sort()
        l1 = img_list[:-1]
        l2 = img_list[1:]
        label = np.zeros([len(l1), 6])
        return l1, l2, label

    def load_data(self):
        samples_list = list(filter(None, re.split('[, ]', self.samples)))  # 提取字符串，并去掉空格
        l1, l2, label = [], [], []
        if self.phase == 'Train':
            # for i in [0, 1, 2, 8, 9]:  # 训练集
            for i in [0]:  # 训练集
                img_list_left = glob(self.dir_data + '/{:02d}/image_2/*.png'.format(i))
                img_list_left.sort()
                img_list_right = glob(self.dir_data + '/{:02d}/image_3/*.png'.format(i))
                img_list_right.sort()
                # i0 左目, i0r 左目反向, ri0 右目, i1 左目隔一帧, i1r 左目隔一帧反向, ri1 右目隔一帧, i2 左目隔两帧, i2r 左目隔两帧反向
                for j in range(len(samples_list)):
                    if samples_list[j] == 'i0':  # 左目
                        l1.extend(img_list_left[:-1])
                        l2.extend(img_list_left[1:])
                        label.extend(np.loadtxt(self.dir_label + '/xyz-euler-relative-interval0/{:d}.txt'.format(i)))
                    elif samples_list[j] == 'i0r':  # 左目反向
                        l1.extend(img_list_left[1:])
                        l2.extend(img_list_left[:-1])
                        label.extend(np.loadtxt(self.dir_label + '/xyz-euler-relative-reverse-interval0/{:d}.txt'.
                                                format(i)))
        # 用序列5进行验证
        else:
            seq_val = 5
            img_list = glob(self.dir_data + '/{:02d}/image_2/*.png'.format(seq_val))
            img_list.sort()
            max_index = int(len(img_list) // 64) * 32
            l1 = img_list[: max_index]
            l2 = img_list[1: max_index + 1]
            label1 = np.loadtxt(self.dir_label + '/xyz-euler-relative-interval0/{:d}.txt'.format(seq_val))
            label = label1[: max_index]
        return l1, l2, label

    def __len__(self):
        return len(self.l1)

    def __getitem__(self, idx):
        """ get one sample
        :param idx: the index of one sample, choose from range(len(self.l1))
        :return: sample: {'img': size[6, H, W], 'label': size[6]}
        """
        sample = dict()
        img1 = np.array(Image.open(self.l1[idx]).resize((1280, 384)))  # - [88.61, 93.70, 92.11]
        img1 = img1.astype(np.float32)
        img2 = np.array(Image.open(self.l2[idx]).resize((1280, 384)))  # - [88.61, 93.70, 92.11]
        img2 = img2.astype(np.float32)
        sample['img1'] = np.transpose(img1, [2, 0, 1])  # HxWx6 TO 6xHxW
        sample['img2'] = np.transpose(img2, [2, 0, 1])  # HxWx6 TO 6xHxW
        label1 = self.label[idx]
        sample['label'] = label1.astype(np.float32)
        return sample


def main():
    dir_data = '/media/czy/DATA/Share/Kitti/color/'
    data_set = KITTIDataSet(dir_data=dir_data,
                            dir_label='.',
                            phase='Train',
                            samples='i0'
                            )

    data_loader = DataLoader(data_set, batch_size=32, shuffle=False, num_workers=4)  # num_workers win上为0 linux上根据配置设置
    print(len(data_set.l1))
    n_batch = int(len(data_set.l1) // data_loader.batch_size)  # 样本总数 / 批大小 = epoch次数
    for i_batch, sample_batch in enumerate(data_loader):
        print(i_batch, n_batch, sample_batch['img1'].size(), sample_batch['img1'].type(), sample_batch['label'].size())


if __name__ == '__main__':
    main()