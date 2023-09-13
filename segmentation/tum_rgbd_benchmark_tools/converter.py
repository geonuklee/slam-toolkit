#-*- coding: utf-8 -*-
#!/usr/bin/python2
# Written by Geonuk LEE, 2023

from scipy.spatial.transform import Rotation as rotation_util
import numpy
import argparse
from os import path as osp

def read_kittiodom_poses(filename):
    """
    KITTI odometrydataset / pose/ $Seq.txt를 associate.read_file_list 처럼 읽음.
    ref: https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats

    KITTI odometry format: Twc (3by4) 행렬의 row major array (e11 e12 e13 e14 e21 e22 .. e33 e34\n)
    TUM rgbd format: Twc에 대응하는 (timestamp tx ty tz qx qy qz qw\n)

    KITTI odometry poses 는 image_0, 즉, left gray camera의 coordinate를 기준으로 제공되며, z축이 전방을 가리킨다.
    (kitti_devkit/readme.txt 참고)
    """
    file = open(filename)
    lines = file.read().split("\n")
    output = dict()
    timestamp = 0.
    output = dict()
    if 'from_dcm' in dir(rotation_util):
        mat2rot = rotation_util.from_dcm
    else:
        mat2rot = rotation_util.from_matrix

    for line in lines:
        if len(line)==0 or line[0]=="#":
            continue
        values = [float(v.strip()) for v in line.split(" ") if v.strip()!=""]
        if len(values) != 12:
            import pdb; pdb.set_trace()
        Rt = numpy.array(values).reshape((3,4))
        rot = mat2rot(Rt[:,:3])
        output[timestamp] = Rt[:,3].tolist() + rot.as_quat().tolist()
        timestamp += 1. # 임의의 timestamp
    return output


