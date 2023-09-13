import os
import cv2
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
return : TODO
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
            # TODO 좌표변환. T_w_c 구현.
            oxts.append(OxtsData(packet, Twi))
    if len(oxts) == 0:
        oxts = None
    return oxts


"""
description: read lidar data given 
input: lidar bin path "path", cam 3D to cam 2D image matrix (4,4), lidar 3D to cam 3D matrix (4,4)
output: valid points in lidar coordinates (PointsNum,4)
"""
def read_velodyne(path, P, vtc_mat,IfReduce=True):
    max_row = 374  # y
    max_col = 1241  # x
    if not os.path.exists(path):
        return None

    lidar = np.fromfile(path, dtype=np.float32).reshape((-1, 4))
    if not IfReduce:
        return lidar

    mask = lidar[:, 0] > 0
    lidar = lidar[mask]
    lidar_copy = np.zeros(shape=lidar.shape)
    lidar_copy[:, :] = lidar[:, :]

    velo_tocam = vtc_mat
    lidar[:, 3] = 1
    lidar = np.matmul(lidar, velo_tocam.T)
    img_pts = np.matmul(lidar, P.T)
    velo_tocam = np.mat(velo_tocam).I
    velo_tocam = np.array(velo_tocam)
    normal = velo_tocam
    normal = normal[0:3, 0:4]
    lidar = np.matmul(lidar, normal.T)
    lidar_copy[:, 0:3] = lidar
    x, y = img_pts[:, 0] / img_pts[:, 2], img_pts[:, 1] / img_pts[:, 2]
    mask = np.logical_and(np.logical_and(x >= 0, x < max_col), np.logical_and(y >= 0, y < max_row))

    return lidar_copy[mask]


"""
description: convert 3D camera coordinates to Lidar 3D coordinates.
input: (PointsNum,3)
output: (PointsNum,3)
"""
def cam_to_velo(cloud,vtc_mat):
    mat=np.ones(shape=(cloud.shape[0],4),dtype=np.float32)
    mat[:,0:3]=cloud[:,0:3]
    mat=np.mat(mat)
    normal=np.mat(vtc_mat).I
    normal=normal[0:3,0:4]
    transformed_mat = normal * mat.T
    T=np.array(transformed_mat.T,dtype=np.float32)
    return T

"""
description: convert 3D camera coordinates to Lidar 3D coordinates.
input: (PointsNum,3)
output: (PointsNum,3)
"""
def velo_to_cam(cloud,vtc_mat):
    mat=np.ones(shape=(cloud.shape[0],4),dtype=np.float32)
    mat[:,0:3]=cloud[:,0:3]
    mat=np.mat(mat)
    normal=np.mat(vtc_mat).I
    normal=normal[0:3,0:4]
    transformed_mat = normal * mat.T
    T=np.array(transformed_mat.T,dtype=np.float32)
    return T

def read_image(path):
    if os.path.exists(path):
        im=cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
    else:
        im=None
    return im

def read_detection_label(path):

    boxes = []
    names = []

    with open(path) as f:
        for line in f.readlines():
            line = line.split()
            this_name = line[0]
            if this_name != "DontCare":
                line = np.array(line[-7:],np.float32)
                boxes.append(line)
                names.append(this_name)

    return np.array(boxes),np.array(names)

def read_tracking_label(path):

    frame_dict={}

    names_dict={}

    with open(path) as f:
        for line in f.readlines():
            line = line.split()
            this_name = line[2]
            frame_id = int(line[0])
            ob_id = int(line[1])

            if this_name != "DontCare":
                line = np.array(line[10:17],np.float32).tolist()
                line.append(ob_id)


                if frame_id in frame_dict.keys():
                    frame_dict[frame_id].append(line)
                    names_dict[frame_id].append(this_name)
                else:
                    frame_dict[frame_id] = [line]
                    names_dict[frame_id] = [this_name]

    return frame_dict,names_dict

if __name__ == '__main__':
    path = 'H:/dataset/traking/training/label_02/0000.txt'
    labels,a = read_tracking_label(path)
    print(a)

