#!/usr/bin/python3
#-*- coding:utf-8 -*-

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow.compat.v1 as tf
import math
import numpy as np
import itertools
tf.enable_eager_execution()

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset
import cv2
import time
import argparse
from os import path as osp
from shutil import rmtree

from depth_map import dense_map
from scipy.spatial.transform import Rotation

def get_uvz_points(range_image_cartesian,
        extrinsic,
        camera_projection,
        camera_name,
        pool_method=tf.math.unsorted_segment_min,
        scale=.5,
        scope=None):
    with tf.compat.v1.name_scope(scope, 'BuildCameraDepthImage',
                    [range_image_cartesian, extrinsic, camera_projection]):
        vehicle_to_camera = tf.linalg.inv(extrinsic) # [B=1,4,4]
        vehicle_to_camera_rotation = vehicle_to_camera[:, 0:3, 0:3] # [B=1,3,3]
        vehicle_to_camera_translation = vehicle_to_camera[:, 0:3, 3] # [B=1,3]
        # xyz(range_image_cartesian, [B,H,W,3]) 의 vehicle->camera 좌표변환
        range_image_camera = tf.einsum(
                        'bij,bhwj->bhwi', vehicle_to_camera_rotation,
                        range_image_cartesian) + vehicle_to_camera_translation[:, tf.newaxis,
                                        tf.newaxis, :] # [B=1,H,W,3]
        # Computing camera_projection_mask
        camera_projection_mask_1 = tf.tile(
                        tf.equal(camera_projection[..., 0:1], camera_name), [1, 1, 1, 2])
        camera_projection_mask_2 = tf.tile(
                        tf.equal(camera_projection[..., 3:4], camera_name), [1, 1, 1, 2])
        camera_projection_selected = tf.ones_like(
                        camera_projection[..., 1:3], dtype=camera_projection.dtype) * -1
        camera_projection_selected = tf.compat.v1.where(camera_projection_mask_2,
                        camera_projection[..., 4:6],
                        camera_projection_selected)
        # [B, H, W, 2]
        camera_projection_selected = tf.compat.v1.where(camera_projection_mask_1,
                        camera_projection[..., 1:3],
                        camera_projection_selected)
        # [B, H, W]
        camera_projection_mask = tf.logical_or(camera_projection_mask_1,
                        camera_projection_mask_2)[..., 0]

    # range image의 norm을 맵핑하는 scatter 대신 dense_map을 호출할 방법이 필요해서,
    # range_image_camera         : 1, 64, 2650, 3
    # range_image_camera_norm    : 1, 64, 2650
    # camera_projection_mask     : 1, 64, 2650
    # camera_projection_selected : 1, 64, 2650, 2

    # 1) waymodataset은 x축을 깊이방향 좌표계로 삼고있었다.
    # 2) 그리고 dense_map의 pts는 u,v,z다 . (x,y,z)가 아니라
    Xc = tf.boolean_mask(range_image_camera, camera_projection_mask).numpy()
    uv = tf.boolean_mask(camera_projection[...,1:3], camera_projection_mask).numpy()
    uv = scale * uv
    uvz_points  = np.hstack([uv,Xc[:,0].reshape(-1,1)]) # Waymodatset은 depth를 x-axis에 할당함.
    return uvz_points

def parse_depth(range_image_cartesian,
                camera_projections,
                image_name,
                camera_image_shape,
                extrinsic,
                use_inpaint,
                verbose,
                depth_compute_scale = .25
                ):
    # depth_compute_scale : desne_depth의 projection계산부담을 줄이기위해 intrinsic의 scale을 낮추는 이미지 배율

    # ric : range image cartesian (reshaped as [batch,w,h,channel] tensor) 
    ric_shape = range_image_cartesian[image_name].shape
    ric = np.reshape(range_image_cartesian[image_name], [1, ric_shape[0], ric_shape[1], ric_shape[2]])

    cp = camera_projections[image_name][0]
    cp_tensor = tf.reshape(tf.convert_to_tensor(value=cp.data), cp.shape.dims)
    cp_shape = cp_tensor.shape
    cp_tensor = np.reshape(cp_tensor, [1, cp_shape[0], cp_shape[1], cp_shape[2]])

    '''
    다음 두 코드가 모두 dense depth map을 획득하는데 부적합해서 선언한 함수의 호출.
    1) range_image_utils.build_camera_depth_image
        : 'range (sqrt(x^2+y^2+z^2))'를 낱개의 image pixel에 맵핑, sparse range image를 만든다.
    2) https://github.com/waymo-research/waymo-open-dataset/blob/master/tutorial/tutorial.ipynb
        : 마찬가지로 'range'를 visualization한 projected points를 rgb 이미지 위해 뿌려 visualization만 수행한다.
    따라서 sqrt(x^2+y^2+z^2) 대신 'z'값을 밀집한 깊이 이미지를 만들어야 한다.
    '''
    # uvz__points (Nx3) : [:,:2]는 projected image point, [:,2]는 cv의 notation을 따르는 camera좌표계에서 z값.
    uvz_points = get_uvz_points(ric, extrinsic, cp_tensor, image_name,scale=depth_compute_scale)
    sw = int(depth_compute_scale*camera_image_shape[1])
    sh = int(depth_compute_scale*camera_image_shape[0])
    depth =  dense_map(uvz_points.T, sw, sh, grid=2).astype(np.float32) # 이게 오래걸린다

    if use_inpaint:
        m0 = ~(depth < 1000.)
        m1 = depth < 0.001
        dist = cv2.distanceTransform( (m1).astype(np.uint8), distanceType=cv2.DIST_L2, maskSize=5)
        m1d = np.logical_and(dist<2.,m1)
        m = np.logical_or(m0,m1d)
        depth = cv2.inpaint(depth,m.astype(np.uint8),3, cv2.INPAINT_TELEA)

    #depth = cv2.resize(depth, (camera_image_shape[1], camera_image_shape[0]), interpolation=cv2.INTER_NEAREST)
    return depth

def InverseTransfrom(R, tvec):
    #return mR.T, -np.matmul(mR.T, tvec)
    return R.inv(), -R.inv().apply(tvec)

def MultTransform(Ra, ta, Rb, tb):
    #return np.matmul(mRa,mRb), np.matmul(mRa,tb)+ta
    return Ra*Rb, Ra.apply(tb)+ta

def Transform2str(R, t):
    Twc = np.hstack((R.as_matrix(),t.reshape(-1,1))).reshape((-1,))
    msg = ''
    for elem in Twc:
        msg += ' %e'%elem
    msg = msg[1:]
    return msg

def parse_tfrecord(filename, use_inpaint=True,verbose=True):
    # Reference : https://github.com/waymo-research/waymo-open-dataset/blob/master/tutorial/tutorial.ipynb
    depth_compute_scale = .25

    dataset = tf.data.TFRecordDataset(filename, compression_type='')
    fdtype = np.float32
    RWw, tWw = None, None
    # RcC : Fix coordinate system as {opencv camera} <- {waymo Camera}
    RcC = np.array( [0., -1., 0.,
                     0., 0., -1.,
                     1., 0., 0.], dtype=fdtype).reshape((3,3))
    RcC = Rotation.from_matrix(RcC)

    output_dir = 'output'
    poses_dir = osp.join(output_dir,'poses')
    sequences_dir = osp.join(output_dir,'sequences')

    if not osp.exists(output_dir):
        os.mkdir(output_dir)
        os.mkdir(poses_dir)
        os.mkdir(sequences_dir)

    seq_name = osp.splitext(osp.basename(filename))[0]
    seq_dir = osp.join(sequences_dir,seq_name)
    pos_fn  = osp.join(poses_dir, seq_name+'.txt')
    if osp.exists(seq_dir):
        rmtree(seq_dir)
    if osp.exists(pos_fn):
        os.remove(pos_fn)
    os.mkdir(seq_dir)
    calib_fn = osp.join(seq_dir, 'calib.txt')
    im_dir   = osp.join(seq_dir, 'image_0')
    depth_dir= osp.join(seq_dir, 'depth_0') # TODO KITTI directory structure for depth?
    os.mkdir(im_dir)
    os.mkdir(depth_dir)
    f_pos = open(pos_fn, 'w')
    i_frame = 0
    for data in dataset:
        t0 = time.time()
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        (range_images, camera_projections, _, range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frame)
        range_image_cartesian = frame_utils.convert_range_image_to_cartesian(frame,
                range_images,
                range_image_top_pose,
                ri_index=0,
                keep_polar_features=False)

        # ref : https://github.com/waymo-research/waymo-open-dataset/blob/master/tutorial/tutorial_camera_only.ipynb
        # Denote : Waymodataset은 차량,카메라의 전방이 'x' axis로 좌표를 할당
        # TWv : {waymo World} <- {vehicle}
        TWv = np.array(frame.pose.transform).reshape(4, 4).astype(fdtype) # Transform into world from vehicle coordinate system.
        RWv, tWv = Rotation.from_matrix(TWv[:3,:3]), TWv[:3,-1]
        RvW, tvW = InverseTransfrom(RWv, tWv)

        ccalib = None
        for im in frame.images:
            camera_name = open_dataset.CameraName.Name.Name(im.name)
            if camera_name != 'FRONT':
                continue
            if ccalib is None:
                ccalib = frame.context.camera_calibrations[im.name-1]
                # ref) https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/dataset.proto#L93
                # calib.intrinsic : 1d Array of [f_u,f_v,c_u,c_v, k1,k2,p1,p2,k3].
                fu,fv,cu,cv,k1,k2,p1,p2,k3 = ccalib.intrinsic
                K0 = np.array([fu, 0.,cu, 0.,fv,cv, 0.,0.,1.]).reshape((3,3))
                D0 = np.array([k1,k2,p1,p2,k3])
                output_scale = .5
                output_size = int(output_scale*ccalib.width), int(output_scale*ccalib.height)
                K, _ = cv2.getOptimalNewCameraMatrix(K0, D0,
                                                    (ccalib.width, ccalib.height), 0.,
                                                    output_size)
                rgb_mapx, rgb_mapy = cv2.initUndistortRectifyMap(K0,D0,None,K,output_size,cv2.CV_32FC1)
                K0[:2,:] *= depth_compute_scale
                depth_mapx, depth_mapy = cv2.initUndistortRectifyMap(K0,D0,None,K,output_size,cv2.CV_32FC1)
                camera_image_shape = (ccalib.height, ccalib.width)

                with open(calib_fn,'w') as f:
                    # TODO intrinsic as newK
                    msg = "P0:"
                    P = np.hstack((K,np.zeros((3,1),dtype=K.dtype))).reshape(-1,).tolist()
                    for p in P:
                        msg += ' %e'%p
                    f.write(msg)

            # TCv : {waymo Camera} <- {vehicle}
            TCv = np.reshape(ccalib.extrinsic.transform,
                    [4,4]).astype(fdtype)
            Rcv, tcv = MultTransform(RcC,np.zeros_like(tWv),
                                    Rotation.from_matrix(TCv[:3,:3]), TCv[:3,-1])

            # TcW = Tcv * TvW : {oepncv camera} <- {waymo World}
            RcW,tcW = MultTransform(Rcv,tcv,RvW,tvW)
            if RWw is None:
                # TWw = TcW.inv() : {waymo World} <- {kitti world}
                RWw, tWw = InverseTransfrom(RcW, tcW)
                Rcw, tcw = Rotation.from_matrix(np.eye(3)), np.zeros_like(tWv)
            else:
                # Tcw = TcW * TWw : {opencv camera} <- {kitti world}
                Rcw, tcw = MultTransform(RcW,tcW, RWw,tWw)

            Rwc, twc = InverseTransfrom(Rcw, tcw)
            msg = Transform2str(Rwc,twc) + "\n"
            # Twc 값을 row major로 풀어서, pos_fn에 저장
            f_pos.write(msg)
            f_pos.flush()
            # Verbose camera translation
            #print('twc = %.3f, %.3f, %.3f' % ( twc[0], twc[1], twc[2]) )

            rgb = tf.image.decode_jpeg(im.image).numpy()
            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            depth = parse_depth(range_image_cartesian,
                                camera_projections,
                                im.name,
                                camera_image_shape,
                                np.reshape(TCv, [1,4,4]),
                                use_inpaint,
                                verbose,
                                depth_compute_scale)
            # Rectification
            rgb = cv2.remap(rgb, rgb_mapx,rgb_mapy,cv2.INTER_NEAREST)
            depth = cv2.remap(depth, depth_mapx,depth_mapy,cv2.INTER_NEAREST)

            # TODO 06d?
            cv2.imwrite(osp.join(im_dir,   '%06d.png'%i_frame), rgb)
            cv2.imwrite(osp.join(depth_dir,'%06d.png'%i_frame), depth)
            i_frame += 1
            if(i_frame >= 1e+6):
                print("Unexpected number of frames.")
                exit(1)

            '''
                TODO
                * [x] Resize depth for given intrinsic
                * [x] Poses for Tcw
                * [x] extrinsic, distortion as np.array
                    * [x] Rectification
                * [x] pos_fn 저장
                * [x] calib.txt 저장
                * [x] image_0, depth_0 저장
                * [x] calib.txt 불러오기
                * [ ] Batch dataset download code ref https://github.com/RalphMao/Waymo-Dataset-Tool
            '''

            if verbose:
                vis_depth = cv2.normalize(depth,None, 0,1,cv2.NORM_MINMAX)*255
                vis_depth = vis_depth.astype(np.uint8)
                cv2.imshow("depth", vis_depth)
                cv2.imshow("rgb",   rgb)
                c = cv2.waitKey(1)
                if(ord('q') == c):
                    exit(1)
                #print("etime = %.2f [sec]"%(time.time()-t0) )
    print("done")
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse tfrecord of waymo as KITTI format')
    parser.add_argument('--filename', help="The filename of the tfrecord",
            default='segment-10203656353524179475_7625_000_7645_000_with_camera_labels.tfrecord') 
    args = parser.parse_args()
    filename = args.filename
    parse_tfrecord(filename)
