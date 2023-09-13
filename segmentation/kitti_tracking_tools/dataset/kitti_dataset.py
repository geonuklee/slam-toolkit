#-*- coding: utf-8 -*-
#/usr/bin/python3

import numpy as np
import re
from .kitti_data_base import *
import os
from scipy.spatial.transform import Rotation as rotation_util

def InvTransform(T):
    R = T[:3,:3].T
    t = - np.matmul(R, T[:3,3])
    Rt = np.concatenate( [R,t.reshape((-1,1))], 1)
    return np.concatenate([ Rt, [[0,0,0,1]] ])

class KittiTrackingDataset:
    def __init__(self,root_path,seq_id,label_path=None):
        self.seq_name = str(seq_id).zfill(4)
        self.root_path = root_path
        self.velo_path = os.path.join(self.root_path,"velodyne",self.seq_name)
        self.lcolor_path = os.path.join(self.root_path,"image_02",self.seq_name)
        self.rcolor_path = os.path.join(self.root_path,"image_03",self.seq_name)
        self.calib_path = os.path.join(self.root_path,"calib",self.seq_name)
        self.oxts_path = os.path.join(self.root_path,"oxts",self.seq_name)
        if os.path.exists(self.velo_path):
            self.all_ids = os.listdir(self.velo_path)
        else:
            self.all_ids = os.listdir(self.lcolor_path)
        self.calib_path = self.calib_path + '.txt'
        self.oxts_path = self.oxts_path + '.txt'
        if label_path is None:
            label_path = os.path.join(self.root_path, "label_02", self.seq_name+'.txt')

        self.P0,self.P1,self.P2,self.P3,\
                T_r_v,T_v_i,T_c0_r,\
                self.V2C = read_calib(self.calib_path)
        self.K0 = self.P0[:,:3] # 사실 넷 모두 intrinsic이 같다.
        self.K1 = self.P1[:,:3]
        self.K2 = self.P2[:,:3]
        self.K3 = self.P3[:,:3]

        '''
        T_w_c2 := T_w_i * T_i_v * T_v_c0 * T_c0_c2
        T_c0_c2 = [ I , -K2.inv()*P2.translation]
        '''
        T_i_v = InvTransform(T_v_i) # TODO Rt
        T_v_c0 = InvTransform(self.V2C)
        t_c0_c2 = -np.matmul(np.linalg.inv(self.K2), self.P2[:3,-1])
        t_c0_c3 = -np.matmul(np.linalg.inv(self.K3), self.P3[:3,-1])
        base_line = - t_c0_c2 + t_c0_c3
        self.base_line = base_line[0] # == t_c2_c3.x()

        T_c0_c2 = np.eye(4, dtype=T_i_v.dtype)
        T_c0_c2[:3,-1] = t_c0_c2
        T_i_c2 = np.matmul(np.matmul(T_i_v,T_v_c0),T_c0_c2)

        self.T_c2_v = InvTransform( np.matmul(T_v_i, T_i_c2) )

        # odometry poses start with Identity, for camera_2's coordinate.
        if 'from_dcm' in dir(rotation_util):
            mat2rot = rotation_util.from_dcm
        else:
            mat2rot = rotation_util.from_matrix
        self.poses = []
        self.oxts = read_oxts(self.oxts_path)
        for packet, T_w_i in self.oxts:
            T_w_c2 = np.matmul(T_w_i, T_i_c2)
            if len(self.poses) == 0:
                T_k0 = InvTransform(T_w_c2)
                T_w_c2 = np.eye(4,dtype=T_k0.dtype)
            else:
                T_w_c2 = np.matmul(T_k0, T_w_c2)
                T_w_c2[-1,:3] = 0.
                T_w_c2[-1,-1] = 1.
            rot = mat2rot(T_w_c2[:3,:3])
            self.poses.append( T_w_c2[:3,-1].tolist() + rot.as_quat().tolist() )
        self.labels, self.label_names = read_tracking_label(label_path)

    def __len__(self):
        return len(self.all_ids)-1

    def __getitem__(self, item):

        name = str(item).zfill(6)

        velo_path = os.path.join(self.velo_path,name+'.bin')
        lcolor_path = os.path.join(self.lcolor_path, name+'.png')
        rcolor_path = os.path.join(self.rcolor_path, name+'.png')
        lcolor = read_image(lcolor_path)
        rcolor = read_image(rcolor_path)
        points = read_velodyne(velo_path,self.P2,self.V2C)

        if item in self.labels.keys():
            labels = self.labels[item]
            labels = np.array(labels)
            labels[:,3:6] = cam_to_velo(labels[:,3:6],self.V2C)[:,:3]
            label_names = self.label_names[item]
            label_names = np.array(label_names)
        else:
            labels = None
            label_names = None

        return points,lcolor,rcolor,self.oxts[item], self.poses[item], labels,label_names
