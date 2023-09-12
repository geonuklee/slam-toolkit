#-*- coding: utf-8 -*-
#/usr/bin/python3

import numpy as np
import re
from .kitti_data_base import *
import os

class KittiTrackingDataset:
    def __init__(self,root_path,seq_id,label_path=None):
        self.seq_name = str(seq_id).zfill(4)
        self.root_path = root_path
        self.velo_path = os.path.join(self.root_path,"velodyne",self.seq_name)
        self.image_path = os.path.join(self.root_path,"image_02",self.seq_name)
        self.calib_path = os.path.join(self.root_path,"calib",self.seq_name)
        if os.path.exists(self.velo_path):
            self.all_ids = os.listdir(self.velo_path)
        else:
            self.all_ids = os.listdir(self.image_path)
        self.calib_path = self.calib_path + '.txt'
        if label_path is None:
            label_path = os.path.join(self.root_path, "label_02", self.seq_name+'.txt')

        self.P2, self.V2C = read_calib(self.calib_path)
        self.labels, self.label_names = read_tracking_label(label_path)

    def __len__(self):
        return len(self.all_ids)-1

    def __getitem__(self, item):

        name = str(item).zfill(6)

        velo_path = os.path.join(self.velo_path,name+'.bin')
        image_path = os.path.join(self.image_path, name+'.png')

        points = read_velodyne(velo_path,self.P2,self.V2C)
        image = read_image(image_path)

        if item in self.labels.keys():
            labels = self.labels[item]
            labels = np.array(labels)
            labels[:,3:6] = cam_to_velo(labels[:,3:6],self.V2C)[:,:3]
            label_names = self.label_names[item]
            label_names = np.array(label_names)
        else:
            labels = None
            label_names = None

        return self.P2,self.V2C,points,image,labels,label_names
