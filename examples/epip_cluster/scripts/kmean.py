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

        Z = img.reshape((-1,3))
        # convert to np.float32
        Z = np.float32(Z)
        # define criteria, number of clusters(K) and apply kmeans()
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret,label,center=cv.kmeans(Z,K,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)
        # Now convert back into uint8, and make original image
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((img.shape))

        edge = cv.Canny(res2, 0,1)

        cv.imshow('res2',res2)
        cv.imshow('edge',edge)

        c = cv.waitKey(1)
        if c == ord('q'):
            break

