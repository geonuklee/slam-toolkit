#!/usr/bin/python
#-*- coding:utf-8 -*-

import numpy as np
import cv2 as cv
from os import path as osp


if __name__ == '__main__':
    seq = '20'
    seq_dir = '/home/geo/dataset/kitti_odometry_dataset/sequences/%s'%seq
    i_img = 0 
    K = 4
    while True:
        im_fn = osp.join(seq_dir, 'image_0/%06d.png'%i_img)
        img = cv.imread(im_fn)
        i_img+= 1
        if img is None:
            break



        cv.imshow('edge',edge)

        c = cv.waitKey(1)
        if c == ord('q'):
            break

