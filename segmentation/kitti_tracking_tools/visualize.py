#-*- coding: utf-8 -*-
#/usr/bin/python3

from viewer.viewer import Viewer
import numpy as np
from dataset.kitti_dataset import *
from os import path as osp
import cv2

def kitti_viewer():
    root="kitti_tracking_dataset/training"
    seq = "0001" # static
    #seq = "0003" # dynamic
    label_path =osp.join(root, "label_02", seq+".txt")
    dataset = KittiTrackingDataset(root,seq,label_path)

    vi = Viewer(box_type="Kitti")
    P2, P3, V2C = dataset.P2, dataset.P3, dataset.V2C

    for i in range(len(dataset)):
        points, image, r_image, oxt, pose, labels, label_names = dataset[i]
        import pdb; pdb.set_trace()

        mask = label_names=="Car"
        if mask is not None:
            labels = labels[mask]
        label_names = label_names[mask]
        if points is not None:
            vi.add_points(points[:,:3],scatter_filed=points[:,2],color_map_name='viridis')
        vi.add_3D_boxes(labels,box_info=label_names)
        vi.add_3D_cars(labels, box_info=label_names)
        vi.add_image(image)
        vi.set_extrinsic_mat(V2C)
        vi.set_intrinsic_mat(P2)
        vi.show_2D() # show 2d first because 3d clear labels.
        vi.show_3D()
        if len(labels) == 0:
            c = cv2.waitKey(1)
        else:
            c = cv2.waitKey(0)
        if c == ord('q'):
            exit(1) 


if __name__ == '__main__':
    kitti_viewer()
