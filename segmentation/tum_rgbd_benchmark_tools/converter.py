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

import re
import numpy as np
from collections import namedtuple
# Per dataformat.txt
OxtsPacket = namedtuple('OxtsPacket',
                        'lat, lon, alt, ' +
                        'roll, pitch, yaw, ' +
                        'vn, ve, vf, vl, vu, ' +
                        'ax, ay, az, af, al, au, ' +
                        'wx, wy, wz, wf, wl, wu, ' +
                        'pos_accuracy, vel_accuracy, ' +
                        'navstat, numsats, ' +
                        'posmode, velmode, orimode')

# Bundle into an easy-to-access structure
OxtsData = namedtuple('OxtsData', 'packet, T_w_imu')

"""
input: calib txt path
return: P2: (4,4) Projection matrix, C0 to C2(left color) image pixel.
        p3: (4,4) Projection matrix, C0 to C3(right color) image pixel.
        vtc_mat: (4,4)  Tcv, from velodyne to C0, 3D velodyne Lidar coordinates to 3D 'left' camera coordinates
"""
def read_calib(calib_path):
    with open(calib_path) as f:
        for line in f.readlines():
            if line[:2] == "P0":
                P = re.split(" ", line.strip())
                P = np.array(P[-12:], np.float32)
                P0 = P.reshape((3, 4))
            if line[:2] == "P1":
                P= re.split(" ", line.strip())
                P= np.array(P[-12:], np.float32)
                P1 = P.reshape((3, 4))
            if line[:2] == "P2":
                P = re.split(" ", line.strip())
                P = np.array(P[-12:], np.float32)
                P2 = P.reshape((3, 4))
            if line[:2] == "P3":
                P = re.split(" ", line.strip())
                P = np.array(P[-12:], np.float32)
                P3 = P.reshape((3, 4))
            if line[:14] == "Tr_velo_to_cam" or line[:11] == "Tr_velo_cam":
                T_r_v = re.split(" ", line.strip())
                T_r_v = np.array(T_r_v[-12:], np.float32)
                T_r_v = T_r_v.reshape((3, 4))
                T_r_v = np.concatenate([T_r_v, [[0, 0, 0, 1]]])
            if line[:14] == "Tr_imu_to_velo" or line[:11] == "Tr_imu_velo":
                T_v_i = re.split(" ", line.strip())
                T_v_i = np.array(T_v_i[-12:], np.float32)
                T_v_i = T_v_i.reshape((3, 4))
                T_v_i = np.concatenate([T_v_i, [[0, 0, 0, 1]]])
            if line[:7] == "R0_rect" or line[:6] == "R_rect":
                T_c0_r = re.split(" ", line.strip())
                T_c0_r = np.array(T_c0_r[-9:], np.float32)
                T_c0_r = T_c0_r.reshape((3, 3))
                T_c0_r = np.concatenate([T_c0_r, [[0], [0], [0]]], -1)
                T_c0_r = np.concatenate([T_c0_r, [[0, 0, 0, 1]]])
    vtc_mat = np.matmul(T_c0_r, T_r_v) # R0 * T_{ref,velo}
    return (P0,P1,P2,P3,T_r_v,T_v_i,T_c0_r,vtc_mat)


def rotx(t):
    """Rotation about the x-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1,  0,  0],
                     [0,  c, -s],
                     [0,  s,  c]])


def roty(t):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                     [0,  1,  0],
                     [-s, 0,  c]])


def rotz(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])


def transform_from_rot_trans(R, t):
    """Transforation matrix from rotation matrix and translation vector."""
    R = R.reshape(3, 3)
    t = t.reshape(3, 1)
    return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))

def pose_from_oxts_packet(packet, scale):
    """Helper method to compute a SE(3) pose matrix from an OXTS packet.
    """
    er = 6378137.  # earth radius (approx.) in meters

    # Use a Mercator projection to get the translation vector
    tx = scale * packet.lon * np.pi * er / 180.
    ty = scale * er * \
        np.log(np.tan((90. + packet.lat) * np.pi / 360.))
    tz = packet.alt
    t = np.array([tx, ty, tz])

    # Use the Euler angles to get the rotation matrix
    Rx = rotx(packet.roll)
    Ry = roty(packet.pitch)
    Rz = rotz(packet.yaw)
    R = Rz.dot(Ry.dot(Rx))

    # Combine the translation and rotation into a homogeneous transform
    return R, t

"""
Generator to read OXTS ground truth data.

Poses are given in an East-North-Up coordinate system 
whose origin is the first GPS position.

Original codes : 
https://github.com/utiasSTARS/pykitti/blob/0.3.1/pykitti/utils.py#L12
https://github.com/utiasSTARS/pykitti/blob/0.3.1/pykitti/utils.py#L85

input: oxts txt path
"""
def read_oxts(oxts_path):
    # Scale for Mercator projection (from first lat value)
    scale = None
    # Origin of the global coordinate system (first GPS position)
    origin = None
    oxts = []

    with open(oxts_path) as f:
        for line in f.readlines():
            line = line.split()
            # Last five entries are flags and counts
            line[:-5] = [float(x) for x in line[:-5]]
            line[-5:] = [int(float(x)) for x in line[-5:]]
            packet = OxtsPacket(*line)
            if scale is None:
                scale = np.cos(packet.lat * np.pi / 180.)
            R, t = pose_from_oxts_packet(packet, scale)
            if origin is None:
                origin = t
            Twi = transform_from_rot_trans(R, t - origin) # T_w_imu
            oxts.append(OxtsData(packet, Twi))
    if len(oxts) == 0:
        oxts = None
    return oxts

def InvTransform(T):
    R = T[:3,:3].T
    t = - np.matmul(R, T[:3,3])
    Rt = np.concatenate( [R,t.reshape((-1,1))], 1)
    return np.concatenate([ Rt, [[0,0,0,1]] ])

def read_kittiraw_oxts(fn_calib, fn_oxts):
    P0,P1,P2,P3,\
        T_r_v,T_v_i,T_c0_r,\
        V2C = read_calib(fn_calib)
    #K0 = P0[:,:3] # 사실 넷 모두 intrinsic이 같다.
    #K1 = P1[:,:3]
    K2 = P2[:,:3]
    K3 = P3[:,:3]
    '''
    T_w_c2 := T_w_i * T_i_v * T_v_c0 * T_c0_c2
    T_c0_c2 = [ I , -K2.inv()*P2.translation]
    '''
    T_i_v = InvTransform(T_v_i) # TODO Rt
    T_v_c0 = InvTransform(V2C)
    t_c0_c2 = -np.matmul(np.linalg.inv(K2), P2[:3,-1])
    t_c0_c3 = -np.matmul(np.linalg.inv(K3), P3[:3,-1])
    base_line = - t_c0_c2 + t_c0_c3
    base_line = base_line[0] # == t_c2_c3.x()

    T_c0_c2 = np.eye(4, dtype=T_i_v.dtype)
    T_c0_c2[:3,-1] = t_c0_c2
    T_i_c2 = np.matmul(np.matmul(T_i_v,T_v_c0),T_c0_c2)

    T_c2_v = InvTransform( np.matmul(T_v_i, T_i_c2) )

    # odometry poses start with Identity, for camera_2's coordinate.
    if 'from_dcm' in dir(rotation_util):
        mat2rot = rotation_util.from_dcm
    else:
        mat2rot = rotation_util.from_matrix
    oxts = read_oxts(fn_oxts)
    timestamp = 0.
    output = dict()
    for packet, T_w_i in oxts:
        T_w_c2 = np.matmul(T_w_i, T_i_c2)
        if len(output) == 0:
            T_k0 = InvTransform(T_w_c2)
            T_w_c2 = np.eye(4,dtype=T_k0.dtype)
        else:
            T_w_c2 = np.matmul(T_k0, T_w_c2)
            T_w_c2[-1,:3] = 0.
            T_w_c2[-1,-1] = 1.
        rot = mat2rot(T_w_c2[:3,:3])
        #poses.append( T_w_c2[:3,-1].tolist() + rot.as_quat().tolist() )
        output[timestamp] = T_w_c2[:3,-1].tolist() + rot.as_quat().tolist()
        timestamp += 1. # 임의의 timestamp
    return output
