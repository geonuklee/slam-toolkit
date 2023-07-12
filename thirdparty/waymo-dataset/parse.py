#!/usr/bin/python3
#-*- coding:utf-8 -*-

'''
    Reference : https://github.com/waymo-research/waymo-open-dataset/blob/master/tutorial/tutorial.ipynb
'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow.compat.v1 as tf
import math
import numpy as np
import itertools
tf.enable_eager_execution()

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

import matplotlib.pyplot as plt
import cv2

def rgba(d):
    """Generates a color based on range.

    Args:
      d: the depth value of a given point.
    Returns:
      The color for a given depth
    """
    c = plt.get_cmap('jet')((d % 20.0) / 20.0)
    c = list(c)
    c[-1] = 0.5  # alpha
    return c

def plot_image(camera_image):
    """Plot a cmaera image."""
    plt.figure(figsize=(20, 12))
    plt.imshow(tf.image.decode_jpeg(camera_image.image))
    plt.grid("off")
    return

def plot_points_on_image(projected_points, camera_image, rgba_func,
        point_size=5.0):
    plot_image(camera_image)
    xs = []
    ys = []
    colors = []
    for point in projected_points:
        xs.append(point[0])  # width, col
        ys.append(point[1])  # height, row
        colors.append(rgba_func(point[2]))
    plt.scatter(xs, ys, c=colors, s=point_size, edgecolors="none")
    return


def func1():
    # Reference : https://github.com/waymo-research/waymo-open-dataset/blob/master/tutorial/tutorial.ipynb
    FILENAME = '/home/geo/ws/slam-toolkit/thirdparty/segment-10203656353524179475_7625_000_7645_000_with_camera_labels.tfrecord'
    dataset = tf.data.TFRecordDataset(FILENAME, compression_type='')
    for data in dataset:
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        (range_images, camera_projections, _, range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frame)
        # points : {[N,3]} 3d lidar points
        # cp_poitns {[N,6]} list of camera projections of length n(lidar)
        points, cp_points = frame_utils.convert_range_image_to_point_cloud(
                frame,
                range_images,
                camera_projections,
                range_image_top_pose)
        points_ri2, cp_points_ri2 = frame_utils.convert_range_image_to_point_cloud(
                frame,
                range_images,
                camera_projections,
                range_image_top_pose,
                ri_index=1)
        # 3d points in vehicle frame.
        points_all = np.concatenate(points, axis=0)
        points_all_ri2 = np.concatenate(points_ri2, axis=0)
        # camera projection corresponding to each point.
        cp_points_all = np.concatenate(cp_points, axis=0)
        cp_points_all_ri2 = np.concatenate(cp_points_ri2, axis=0)
        images = sorted(frame.images, key=lambda i:i.name)
        cp_points_all_concat = np.concatenate([cp_points_all, points_all], axis=-1)
        cp_points_all_concat_tensor = tf.constant(cp_points_all_concat)
        _cp_points_all_tensor = tf.constant(cp_points_all, dtype=tf.int32)

        # range_all_tensor = tf.norm(points_all, axis=-1, keepdims=True)
        for index, image in enumerate(frame.images): 
            # frame.images.name != frame.context.lidars.name 임을 주의.
            # ref : https://github.com/waymo-research/waymo-open-dataset/blob/656f759070a7b1356f9f0403b17cd85323e0626c/src/waymo_open_dataset/dataset.proto
            str_name = open_dataset.CameraName.Name.Name(image.name)
            if str_name != 'FRONT':
                continue
            mask = tf.equal(_cp_points_all_tensor[..., 0], image.name)
            cp_points_all_tensor = tf.cast(tf.gather_nd(
                _cp_points_all_tensor, tf.where(mask)), dtype=tf.float32)
            #range_all_tensor = tf.gather_nd(range_all_tensor, tf.where(mask))
            # TODO points_all은 vechile frame이지 camera frame이 아니다...
            # ref : https://github.com/waymo-research/waymo-open-dataset/blob/656f759070a7b1356f9f0403b17cd85323e0626c/src/waymo_open_dataset/utils/range_image_utils.py#L388
            depth_all_tensor = cp_points_all_tensor[...,3:4]

            # cp_points_all_tensor의 1,2 (1:2)는 는 각각 projected image u, v
            projected_points_all_from_raw_data \
                    = tf.concat([cp_points_all_tensor[..., 1:3], depth_all_tensor], axis=-1).numpy()
            plot_points_on_image(projected_points_all_from_raw_data, image, rgba, point_size=5.0)
            plt.show()
        break # Break data

# From Github https://github.com/balcilar/DenseDepthMap
def dense_map(Pts, n, m, grid):
    ng = 2 * grid + 1
    mX = np.zeros((m,n)) + np.float("inf")
    mY = np.zeros((m,n)) + np.float("inf")
    mD = np.zeros((m,n))
    mX[np.int32(Pts[1]),np.int32(Pts[0])] = Pts[0] - np.round(Pts[0])
    mY[np.int32(Pts[1]),np.int32(Pts[0])] = Pts[1] - np.round(Pts[1])
    mD[np.int32(Pts[1]),np.int32(Pts[0])] = Pts[2]
    KmX = np.zeros((ng, ng, m - ng, n - ng))
    KmY = np.zeros((ng, ng, m - ng, n - ng))
    KmD = np.zeros((ng, ng, m - ng, n - ng))
    for i in range(ng):
        for j in range(ng):
            KmX[i,j] = mX[i : (m - ng + i), j : (n - ng + j)] - grid - 1 +i
            KmY[i,j] = mY[i : (m - ng + i), j : (n - ng + j)] - grid - 1 +i
            KmD[i,j] = mD[i : (m - ng + i), j : (n - ng + j)]
    S = np.zeros_like(KmD[0,0])
    Y = np.zeros_like(KmD[0,0])
    for i in range(ng):
        for j in range(ng):
            s = 1/np.sqrt(KmX[i,j] * KmX[i,j] + KmY[i,j] * KmY[i,j])
            Y = Y + s * KmD[i,j]
            S = S + s
    S[S == 0] = 1
    out = np.zeros((m,n))
    out[grid + 1 : -grid, grid + 1 : -grid] = Y/S
    return out

def get_camera_points(range_image_cartesian,
        extrinsic,
        camera_projection,
        camera_image_size,
        camera_name,
        pool_method=tf.math.unsorted_segment_min,
        scale=.5,
        scope=None):
    # ref : range_image_utils.build_camera_depth_image
    '''
    Args:
      range_image_cartesian: [B, H, W, 3] tensor. Range image points in vehicle
        frame. Note that if the range image is provided by pixel_pose, then you
        can optionally pass in the cartesian coordinates in each pixel frame.
      extrinsic: [B, 4, 4] tensor. Camera extrinsic.
      camera_projection: [B, H, W, 6] tensor. Each range image pixel is associated
        with at most two camera projections. See dataset.proto for more details.
      camera_image_size: a list of [height, width] integers.
      camera_name: an integer that identifies a camera. See dataset.proto.
      pool_method: pooling method when multiple lidar points are projected to one
        image pixel.
      scope: the name scope.

    Returns:
    TODO
    '''
    with tf.compat.v1.name_scope(scope, 'BuildCameraDepthImage',
                    [range_image_cartesian, extrinsic, camera_projection]):
        # [B, 4, 4]
        vehicle_to_camera = tf.linalg.inv(extrinsic)
        # [B, 3, 3]
        vehicle_to_camera_rotation = vehicle_to_camera[:, 0:3, 0:3]
        # [B, 3]
        vehicle_to_camera_translation = vehicle_to_camera[:, 0:3, 3]
        # [B, H, W, 3]
        # 여기가 xyz(range_image_cartesian, [B,H,W,3]) 의 vehicle->camera 좌표변환
        range_image_camera = tf.einsum(
                        'bij,bhwj->bhwi', vehicle_to_camera_rotation,
                        range_image_cartesian) + vehicle_to_camera_translation[:, tf.newaxis,
                                        tf.newaxis, :]
        # [B, H, W]
        range_image_camera_norm = tf.norm(tensor=range_image_camera, axis=-1)
        # TODO norm 대신 [...,2]

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

        def fn(args):
            """Builds depth image for a single frame."""
            # NOTE: Do not use ri_range > 0 as mask as missing range image pixels are
            # not necessarily populated as range = 0.
            mask, ri_range, cp = args
            mask_ids = tf.compat.v1.where(mask)
            index = tf.gather_nd(
                            tf.stack([cp[..., 1], cp[..., 0]], axis=-1), mask_ids)
            value = tf.gather_nd(ri_range, mask_ids)
            return range_image_utils.scatter_nd_with_pool(index, value, camera_image_size, pool_method)

    # range image의 norm을 맵핑하는 scatter 대신 dense_map을 호출할 방법이 필요하다... 
    # TODO -> 그러므로 'xyz_cam' points를 구하는 함수필요. scatter pool 거창한 결과물 말고
    # 그러고나서 dense_map(pts, width,height,grid=8) 호출하면 끝
    #import pdb; pdb.set_trace()
    # range_image_camera      : 1, 64, 2650, 3
    # range_image_camera_norm : 1, 64, 2650
    # camera_projection_mask  : 1, 64, 2650
    # camera_projection_selected : 1, 64, 2650, 2
    # 1) waymodataset은 x축을 깊이방향 좌표계로 삼고있었다. notion 참고!
    # 그리고 dense_map의 pts는 u,v,z다 . (x,y,z)가 아니라
    pts0 = tf.boolean_mask(range_image_camera, camera_projection_mask).numpy()
    uv = tf.boolean_mask(camera_projection[...,1:3], camera_projection_mask).numpy()
    uv = scale * uv
    width = int(scale*camera_image_size[1])
    height = int(scale*camera_image_size[0])
    pts  = np.hstack([uv,pts0[:,0].reshape(-1,1)])

    # 이게 오래걸린다
    depth =  dense_map(pts.T, width, height, grid=2)
    #depth = tf.map_fn(fn,
    #        elems=[camera_projection_mask, range_image_camera_norm, camera_projection_selected],
    #        dtype=range_image_camera_norm.dtype,
    #        back_prop=False).numpy()[0,:,:]
    #import pdb; pdb.set_trace()
    return depth 



def func2():
    # build depth image
    # ref1 for waymo : https://github.com/waymo-research/waymo-open-dataset/issues/650
    # ref2 for dense depth : https://github.com/BerensRWU/DenseMap/blob/main/depth_map.py
    FILENAME = '/home/geo/ws/slam-toolkit/thirdparty/segment-10203656353524179475_7625_000_7645_000_with_camera_labels.tfrecord'
    dataset = tf.data.TFRecordDataset(FILENAME, compression_type='')
    for data in dataset:
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        (range_images, camera_projections, _, range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frame)
        range_image_cartesian = frame_utils.convert_range_image_to_cartesian(frame,
                range_images,
                range_image_top_pose,
                ri_index=0,
                keep_polar_features=False)
        for im in frame.images:
            camera_name = open_dataset.CameraName.Name.Name(im.name)
            if camera_name != 'FRONT':
                continue
            extrinsic = np.reshape(frame.context.camera_calibrations[im.name-1].extrinsic.transform, [1,4,4]).astype(np.float32)
            camera_image_size = (frame.context.camera_calibrations[im.name-1].height, frame.context.camera_calibrations[im.name-1].width)                
            ric_shape = range_image_cartesian[im.name].shape
            ric = np.reshape(range_image_cartesian[im.name], [1, ric_shape[0], ric_shape[1], ric_shape[2]])

            cp = camera_projections[im.name][0]
            cp_tensor = tf.reshape(tf.convert_to_tensor(value=cp.data), cp.shape.dims)
            cp_shape = cp_tensor.shape
            cp_tensor = np.reshape(cp_tensor, [1, cp_shape[0], cp_shape[1], cp_shape[2]])

            depth = get_camera_points(ric, extrinsic, cp_tensor, camera_image_size, im.name,scale=.25)
            cv_im = tf.image.decode_jpeg(im.image).numpy()
            cv_im = cv2.cvtColor(cv_im, cv2.COLOR_RGB2BGR)
            cv_im = cv2.resize(cv_im, (640,480) )
            ## TODO dense map함수 선언 : https://github.com/BerensRWU/DenseMap/blob/main/depth_map.py
            #depth = range_image_utils.build_camera_depth_image(ric,
            #             extrinsic,
            #             cp_tensor,
            #             camera_image_size,
            #             im.name).numpy()[0,:,:]
            #depth = cv2.pyrDown(depth)
            ndepth = cv2.normalize(depth,None, 0,1,cv2.NORM_MINMAX)*255
            ndepth = ndepth.astype(np.uint8)
            cv2.imshow("depth", ndepth)
            cv2.imshow("rgb", cv_im)
            c = cv2.waitKey(1)
            if(ord('q') == c):
                exit(1)
    print("done")
    cv2.waitKey()
    return

if __name__ == '__main__':
    #func1()
    func2()
