# encoding: utf-8
"""
@author:  l1aoxingyu
@contact: sherlockliao01@gmail.com
"""

import glob
import os
import os.path as osp
import re
import warnings
import numpy as np

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY

@DATASET_REGISTRY.register()
class MSMT17(ImageDataset):

    dataset_dir = ''
    dataset_url = None
    dataset_name = 'msmt17'

    def __init__(self, root='datasets', **kwargs):
        self.dataset_dir = root

        self.data_dir = self.dataset_dir
        data_dir = osp.join(self.data_dir, 'msmt')
        if osp.isdir(data_dir):
            self.data_dir = data_dir
        assert 'Dataset folder not found'

        self.train_dir = osp.join(self.data_dir, 'bounding_box_train_sct')
        self.query_dir = osp.join(self.data_dir, 'query')
        self.gallery_dir = osp.join(self.data_dir, 'bounding_box_test')

        required_files = [
            self.data_dir,
            self.train_dir,
            self.query_dir,
            self.gallery_dir,
        ]

        self.check_before_run(required_files)

        #train = self.process_dir_train(self.train_dir)
        train = self.process_dir_train(self.train_dir)
        query = self.process_dir_test(self.query_dir, is_train=False)
        gallery = self.process_dir_test(self.gallery_dir, is_train=False)

        super(MSMT17, self).__init__(train, query, gallery, **kwargs)


    def process_dir_train(self, dir_path, is_train=True):

        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        data = []
        time_dir = {'morning': 0, 'afternoon': 1, 'noon': 2}
        for img_path in img_paths:
            image_name = os.path.split(img_path)[1]
            pid = int(image_name.split('_')[0])
            camid = int(image_name.split('_')[2]) - 1
            time = image_name.split('_')[3][4:]
            camid = camid*3 + time_dir[time]
            if pid == -1:
                continue  # junk images are just ignored
            if is_train:
                pid = self.dataset_name + "_" + str(pid)
                camid = self.dataset_name + "_" + str(camid)
            data.append((img_path, pid, camid, 0))

        return data

    def process_dir_test(self, dir_path, is_train=True):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        data = []
        for img_path in img_paths:
            image_name = os.path.split(img_path)[1]
            pid = int(image_name.split('_')[0])
            camid = int(image_name.split('_')[1][1:])-1
            if pid == -1:
                continue  # junk images are just ignored
            if is_train:
                pid = self.dataset_name + "_" + str(pid)
                camid = self.dataset_name + "_" + str(camid)

            data.append((img_path, pid, camid, 0))

        return data

