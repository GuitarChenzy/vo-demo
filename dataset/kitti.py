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



    def __len__(self):
        return len(self.l1)

    def __getitem__(self, index) :
        
        return super().__getitem__(index)