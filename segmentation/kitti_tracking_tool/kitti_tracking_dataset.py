#-*- coding:utf-8 -*-

import csv
import os
import re
import cv2
from collections import namedtuple
import numpy as np
from scipy.spatial.transform import Rotation as rotation_util
from scipy.optimize import linear_sum_assignment

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

def read_image(path):
    if os.path.exists(path):
        im=cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
    else:
        im=None
    return im

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

def InvTransform(T):
    R = T[:3,:3].T
    t = - np.matmul(R, T[:3,3])
    Rt = np.concatenate( [R,t.reshape((-1,1))], 1)
    return np.concatenate([ Rt, [[0,0,0,1]] ])

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

from trackeval.datasets import _BaseDataset, Kitti2DBox
from trackeval import utils, _timing
from trackeval.utils import TrackEvalException


class KittiTrackingDataset(_BaseDataset):

    @staticmethod
    def get_default_dataset_config():
        config = Kitti2DBox.get_default_dataset_config()
        config['CLASSES_TO_EVAL'] = ['dynamic']
        return config

    def _get_dynamic_objects(self, label_path, seq, read_data):
        self.lcolor_path = os.path.join(self.gt_fol,"image_02",seq)
        self.rcolor_path = os.path.join(self.gt_fol,"image_03",seq)
        calib_path  = os.path.join(self.gt_fol,"calib",seq)
        oxts_path   = os.path.join(self.gt_fol,"oxts",seq)
        calib_path = calib_path + '.txt'
        oxts_path = oxts_path + '.txt'
        self.P0,self.P1,self.P2,self.P3,\
                T_r_v,T_v_i,T_c0_r,\
                self.V2C = read_calib(calib_path)
        self.K0 = self.P0[:,:3] # 사실 넷 모두 intrinsic이 같다.
        self.K1 = self.P1[:,:3]
        self.K2 = self.P2[:,:3]
        self.K3 = self.P3[:,:3]
        '''
        T_w_c2 := T_w_i * T_i_v * T_v_c0 * T_c0_c2
        T_c0_c2 = [ I , -K2.inv()*P2.translation]
        '''
        T_i_v = InvTransform(T_v_i)
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
        self.oxts = read_oxts(oxts_path)
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

        d_read_data = {}
        X0_for_id = {}
        dynamic_objs = set()
        for k, obj_list in read_data.items():
            if k not in d_read_data:
                d_read_data[k]  = []
            #dst = cv2.imread(os.path.join(self.lcolor_path, k.zfill(6)+'.png'))
            t = int(k)
            pose = self.poses[t]
            T_w_c2 = np.eye(4)
            T_w_c2[:3,:3] = rotation_util.from_quat( pose[3:] ).as_matrix()
            T_w_c2[:3,-1] = pose[:3]
            for obj in obj_list:
                obj_id, org_cls = obj[1], obj[2]
                hwl, xyz, ry = obj[10:13], obj[13:16], obj[16]
                """ KITTI tracking dataset - devkit/readme.md
                    frame trackid cls truncated occluded alpha bbox(4) h,w,l(3) x,y,z(3) rotationy score
                    0     1       2   3         4        5     6:10    10:13    13:16    16        17
                    The reference point for the 3D bounding box for each object is centered on the
                    bottom face of the box. The corners of bounding box are computed as follows with
                    respect to the reference point and in the object coordinate system:
                    x_corners = [l/2, l/2, -l/2, -l/2,  l/2,  l/2, -l/2, -l/2]^T
                    y_corners = [0,   0,    0,    0,   -h,   -h,   -h,   -h  ]^T
                    z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2  ]^T
                """
                Xc = np.array(xyz+[1.], np.float)
                Xw = np.matmul(T_w_c2, Xc)
                if obj_id in X0_for_id:
                    t_err = np.linalg.norm(X0_for_id[obj_id] - Xw)
                else:
                    t_err = 0.
                    X0_for_id[obj_id] = Xw
                if t_err > 2.:
                    dynamic_objs.add(obj_id)
                if obj_id in dynamic_objs:
                    obj[2] = 1 # TODO convert_filter 참고.
                    d_read_data[k].append( obj )
                #bbox = tuple( [int(float(i)) for i in obj[6:10]] )
                #if obj_id in dynamic_objs:
                #    cv2.rectangle(dst, bbox[:2], bbox[2:], (255,0,0), 2)
                #else:
                #    cv2.rectangle(dst, bbox[:2], bbox[2:], (100,100,100), 1)
                #msg = '%s' % obj_id
                #font, fs, ft = cv2.FONT_HERSHEY_PLAIN, 2., 1
                #size, _ = cv2.getTextSize(msg,font,fs,ft)
                #cv2.rectangle(dst, (bbox[0], bbox[1]), (bbox[0]+size[0], bbox[1]-size[1]), (255,255,255),-1)
                #cv2.putText(dst, msg, (bbox[0], bbox[1]),font,fs,(0,0,255),ft)
            if len(d_read_data[k]) == 0:
                d_read_data.pop(k)
            #cv2.imshow("dst",dst)
            #c = cv2.waitKey()
            #if c==ord('q'):
            #    break
        return d_read_data, {} 

    def _load_raw_file(self, tracker, seq, is_gt):
        #return Kitti2DBox._load_raw_file(self, tracker, seq, is_gt)
        """Load a file (gt or tracker) in the kitti 2D box format

        If is_gt, this returns a dict which contains the fields:
        [gt_ids, gt_classes] : list (for each timestep) of 1D NDArrays (for each det).
        [gt_dets, gt_crowd_ignore_regions]: list (for each timestep) of lists of detections.
        [gt_extras] : list (for each timestep) of dicts (for each extra) of 1D NDArrays (for each det).

        if not is_gt, this returns a dict which contains the fields:
        [tracker_ids, tracker_classes, tracker_confidences] : list (for each timestep) of 1D NDArrays (for each det).
        [tracker_dets]: list (for each timestep) of lists of detections.
        """
        # File location
        zip_file = None
        if is_gt:
            file = os.path.join(self.gt_fol, 'label_02', seq + '.txt')
        else:
            file = os.path.join(self.tracker_fol, tracker, self.tracker_sub_fol, seq + '.txt')

        # Ignore regions
        if is_gt:
            crowd_ignore_filter = {2: ['dontcare']}
        else:
            crowd_ignore_filter = None

        # Valid classes
        valid_filter = {2: [x for x in self.class_list]}
        if is_gt:
            if 'car' in self.class_list:
                valid_filter[2].append('van')
            if 'pedestrian' in self.class_list:
                valid_filter[2] += ['person']

        convert_filter = {}
        if is_gt:
            convert_filter[2] =  {'car': 1, 'van': 1, 'truck': 1, 'pedestrian': 1, 'person': 1,
                                  'cyclist': 1, 'tram': 1, 'misc': 1, 'dontcare': 1, 'car_2': 1}
        else:
            convert_filter[2] = {'dynamic':1}

        # Load raw data from text file
        read_data, ignore_data = self._load_simple_text_file(file, time_col=0, id_col=1, remove_negative_ids=True,
                                                             valid_filter=valid_filter,
                                                             crowd_ignore_filter=crowd_ignore_filter,
                                                             convert_filter=convert_filter,
                                                             is_zipped=self.data_is_zipped, zip_file=zip_file)
        if is_gt:
            read_data, ignore_data = self._get_dynamic_objects(file, seq, read_data)
            #import pdb; pdb.set_trace()

        # Convert data to required format
        num_timesteps = self.seq_lengths[seq]
        data_keys = ['ids', 'classes', 'dets']
        if is_gt:
            data_keys += ['gt_crowd_ignore_regions', 'gt_extras']
        else:
            data_keys += ['tracker_confidences']
        raw_data = {key: [None] * num_timesteps for key in data_keys}

        # Check for any extra time keys
        current_time_keys = [str(t) for t in range(num_timesteps)]
        extra_time_keys = [x for x in read_data.keys() if x not in current_time_keys]
        if len(extra_time_keys) > 0:
            if is_gt:
                text = 'Ground-truth'
            else:
                text = 'Tracking'
            raise TrackEvalException(
                text + ' data contains the following invalid timesteps in seq %s: ' % seq + ', '.join(
                    [str(x) + ', ' for x in extra_time_keys]))

        for t in range(num_timesteps):
            time_key = str(t)
            if time_key in read_data.keys():
                time_data = np.asarray(read_data[time_key], dtype=np.float)
                raw_data['dets'][t] = np.atleast_2d(time_data[:, 6:10])
                raw_data['ids'][t] = np.atleast_1d(time_data[:, 1]).astype(int)
                raw_data['classes'][t] = np.atleast_1d(time_data[:, 2]).astype(int)
                if is_gt:
                    gt_extras_dict = {'truncation': np.atleast_1d(time_data[:, 3].astype(int)),
                                      'occlusion': np.atleast_1d(time_data[:, 4].astype(int))}
                    raw_data['gt_extras'][t] = gt_extras_dict
                else:
                    if time_data.shape[1] > 17:
                        raw_data['tracker_confidences'][t] = np.atleast_1d(time_data[:, 17])
                    else:
                        raw_data['tracker_confidences'][t] = np.ones(time_data.shape[0])
            else:
                raw_data['dets'][t] = np.empty((0, 4))
                raw_data['ids'][t] = np.empty(0).astype(int)
                raw_data['classes'][t] = np.empty(0).astype(int)
                if is_gt:
                    gt_extras_dict = {'truncation': np.empty(0),
                                      'occlusion': np.empty(0)}
                    raw_data['gt_extras'][t] = gt_extras_dict
                else:
                    raw_data['tracker_confidences'][t] = np.empty(0)
            if is_gt:
                if time_key in ignore_data.keys():
                    time_ignore = np.asarray(ignore_data[time_key], dtype=np.float)
                    raw_data['gt_crowd_ignore_regions'][t] = np.atleast_2d(time_ignore[:, 6:10])
                else:
                    raw_data['gt_crowd_ignore_regions'][t] = np.empty((0, 4))

        if is_gt:
            key_map = {'ids': 'gt_ids',
                       'classes': 'gt_classes',
                       'dets': 'gt_dets'}
        else:
            key_map = {'ids': 'tracker_ids',
                       'classes': 'tracker_classes',
                       'dets': 'tracker_dets'}
        for k, v in key_map.items():
            raw_data[v] = raw_data.pop(k)
        raw_data['num_timesteps'] = num_timesteps
        raw_data['seq'] = seq
        return raw_data

    @_timing.time
    def get_preprocessed_seq_data(self, raw_data, cls):
        #return Kitti2DBox.get_preprocessed_seq_data(self, raw_data, cls)
        """
        다음 eval_sequence 계산에서 사용되는 data(gt와 tracker instance의 iou 계산 등)를 반환하는 함수.

        ``` trackeval/eval.py/eval_sequence
        # _BaseDataset.get_raw_seq_data()에서 KittiTrackingDataset._load_raw_file(tracker, seq) 결과물에 similarity_scores추가.
        raw_data = dataset.get_raw_seq_data(tracker, seq)

        seq_res = {}
        for cls in class_list:
            seq_res[cls] = {}
            data = dataset.get_preprocessed_seq_data(raw_data, cls)  # <- 이것
            for metric, met_name in zip(metrics_list, metric_names):
                seq_res[cls][met_name] = metric.eval_sequence(data)
        ```

        Kitti2DBox에 하드코딩된 cls 조건문 때문에 아래와 같이 재정의.
        https://github.com/JonathonLuiten/TrackEval/blob/master/trackeval/datasets/kitti_2d_box.py#L238
        """
        #if cls == 'pedestrian':
        #    distractor_classes = [self.class_name_to_class_id['person']]
        #elif cls == 'car':
        #    distractor_classes = [self.class_name_to_class_id['van']]
        if cls == 'dynamic':
            distractor_classes = [1]
        else:
            raise (TrackEvalException('Class %s is not evaluatable' % cls))

        cls_id = self.class_name_to_class_id[cls]

        data_keys = ['gt_ids', 'tracker_ids', 'gt_dets', 'tracker_dets', 'tracker_confidences', 'similarity_scores']
        data = {key: [None] * raw_data['num_timesteps'] for key in data_keys}
        unique_gt_ids = []
        unique_tracker_ids = []
        num_gt_dets = 0
        num_tracker_dets = 0
        for t in range(raw_data['num_timesteps']):

            # Only extract relevant dets for this class for preproc and eval (cls + distractor classes)
            gt_class_mask = np.sum([raw_data['gt_classes'][t] == c for c in [cls_id] + distractor_classes], axis=0)
            gt_class_mask = gt_class_mask.astype(np.bool)
            gt_ids = raw_data['gt_ids'][t][gt_class_mask]
            gt_dets = raw_data['gt_dets'][t][gt_class_mask]
            gt_classes = raw_data['gt_classes'][t][gt_class_mask]
            gt_occlusion = raw_data['gt_extras'][t]['occlusion'][gt_class_mask]
            gt_truncation = raw_data['gt_extras'][t]['truncation'][gt_class_mask]

            tracker_class_mask = np.atleast_1d(raw_data['tracker_classes'][t] == cls_id)
            tracker_class_mask = tracker_class_mask.astype(np.bool)
            tracker_ids = raw_data['tracker_ids'][t][tracker_class_mask]
            tracker_dets = raw_data['tracker_dets'][t][tracker_class_mask]
            tracker_confidences = raw_data['tracker_confidences'][t][tracker_class_mask]
            similarity_scores = raw_data['similarity_scores'][t][gt_class_mask, :][:, tracker_class_mask]

            # Match tracker and gt dets (with hungarian algorithm) and remove tracker dets which match with gt dets
            # which are labeled as truncated, occluded, or belonging to a distractor class.
            to_remove_matched = np.array([], np.int)
            unmatched_indices = np.arange(tracker_ids.shape[0])
            if gt_ids.shape[0] > 0 and tracker_ids.shape[0] > 0:
                matching_scores = similarity_scores.copy()
                matching_scores[matching_scores < 0.5 - np.finfo('float').eps] = 0
                match_rows, match_cols = linear_sum_assignment(-matching_scores)
                actually_matched_mask = matching_scores[match_rows, match_cols] > 0 + np.finfo('float').eps
                match_rows = match_rows[actually_matched_mask]
                match_cols = match_cols[actually_matched_mask]

                is_distractor_class = np.isin(gt_classes[match_rows], distractor_classes)
                is_occluded_or_truncated = np.logical_or(
                    gt_occlusion[match_rows] > self.max_occlusion + np.finfo('float').eps,
                    gt_truncation[match_rows] > self.max_truncation + np.finfo('float').eps)
                to_remove_matched = np.logical_or(is_distractor_class, is_occluded_or_truncated)
                to_remove_matched = match_cols[to_remove_matched]
                unmatched_indices = np.delete(unmatched_indices, match_cols, axis=0)

            # For unmatched tracker dets, also remove those smaller than a minimum height.
            unmatched_tracker_dets = tracker_dets[unmatched_indices, :]
            unmatched_heights = unmatched_tracker_dets[:, 3] - unmatched_tracker_dets[:, 1]
            is_too_small = unmatched_heights <= self.min_height + np.finfo('float').eps

            # For unmatched tracker dets, also remove those that are greater than 50% within a crowd ignore region.
            crowd_ignore_regions = raw_data['gt_crowd_ignore_regions'][t]
            intersection_with_ignore_region = self._calculate_box_ious(unmatched_tracker_dets, crowd_ignore_regions,
                                                                       box_format='x0y0x1y1', do_ioa=True)
            is_within_crowd_ignore_region = np.any(intersection_with_ignore_region > 0.5 + np.finfo('float').eps, axis=1)

            # Apply preprocessing to remove all unwanted tracker dets.
            to_remove_unmatched = unmatched_indices[np.logical_or(is_too_small, is_within_crowd_ignore_region)]
            to_remove_tracker = np.concatenate((to_remove_matched, to_remove_unmatched), axis=0)
            data['tracker_ids'][t] = np.delete(tracker_ids, to_remove_tracker, axis=0)
            data['tracker_dets'][t] = np.delete(tracker_dets, to_remove_tracker, axis=0)
            data['tracker_confidences'][t] = np.delete(tracker_confidences, to_remove_tracker, axis=0)
            similarity_scores = np.delete(similarity_scores, to_remove_tracker, axis=1)

            # Also remove gt dets that were only useful for preprocessing and are not needed for evaluation.
            # These are those that are occluded, truncated and from distractor objects.
            gt_to_keep_mask = (np.less_equal(gt_occlusion, self.max_occlusion)) & \
                              (np.less_equal(gt_truncation, self.max_truncation)) & \
                              (np.equal(gt_classes, cls_id))
            data['gt_ids'][t] = gt_ids[gt_to_keep_mask]
            data['gt_dets'][t] = gt_dets[gt_to_keep_mask, :]
            data['similarity_scores'][t] = similarity_scores[gt_to_keep_mask]

            unique_gt_ids += list(np.unique(data['gt_ids'][t]))
            unique_tracker_ids += list(np.unique(data['tracker_ids'][t]))
            num_tracker_dets += len(data['tracker_ids'][t])
            num_gt_dets += len(data['gt_ids'][t])

        # Re-label IDs such that there are no empty IDs
        if len(unique_gt_ids) > 0:
            unique_gt_ids = np.unique(unique_gt_ids)
            gt_id_map = np.nan * np.ones((np.max(unique_gt_ids) + 1))
            gt_id_map[unique_gt_ids] = np.arange(len(unique_gt_ids))
            for t in range(raw_data['num_timesteps']):
                if len(data['gt_ids'][t]) > 0:
                    data['gt_ids'][t] = gt_id_map[data['gt_ids'][t]].astype(np.int)
        if len(unique_tracker_ids) > 0:
            unique_tracker_ids = np.unique(unique_tracker_ids)
            tracker_id_map = np.nan * np.ones((np.max(unique_tracker_ids) + 1))
            tracker_id_map[unique_tracker_ids] = np.arange(len(unique_tracker_ids))
            for t in range(raw_data['num_timesteps']):
                if len(data['tracker_ids'][t]) > 0:
                    data['tracker_ids'][t] = tracker_id_map[data['tracker_ids'][t]].astype(np.int)

        # Record overview statistics.
        data['num_tracker_dets'] = num_tracker_dets
        data['num_gt_dets'] = num_gt_dets
        data['num_tracker_ids'] = len(unique_tracker_ids)
        data['num_gt_ids'] = len(unique_gt_ids)
        data['num_timesteps'] = raw_data['num_timesteps']
        data['seq'] = raw_data['seq']

        # Ensure that ids are unique per timestep.
        self._check_unique_ids(data)

        return data

    def _calculate_similarities(self, gt_dets_t, tracker_dets_t):
        return Kitti2DBox._calculate_similarities(self, gt_dets_t, tracker_dets_t)

    def __init__(self, config):
        """Initialise dataset, checking that all required files are present"""
        super().__init__()

        """
        TrackEval의 형태로 parsing하면서,
        oxts(gps/imu) parsing도 추가.
        """
        self.config = utils.init_config(config, self.get_default_dataset_config(), self.get_name())
        self.gt_fol = self.config['GT_FOLDER']
        self.tracker_fol = self.config['TRACKERS_FOLDER']
        self.should_classes_combine = False
        self.use_super_categories = False
        self.data_is_zipped = False #self.config['INPUT_AS_ZIP']
        self.output_fol = self.config['OUTPUT_FOLDER']
        if self.output_fol is None:
            self.output_fol = self.tracker_fol
        self.tracker_sub_fol = self.config['TRACKER_SUB_FOLDER']
        self.output_sub_fol = self.config['OUTPUT_SUB_FOLDER']
        self.max_occlusion = 2
        self.max_truncation = 0
        self.min_height = 25
        # Get classes to eval
        self.valid_classes = ['dynamic']
        self.class_list = [cls.lower() if cls.lower() in self.valid_classes else None
                           for cls in self.config['CLASSES_TO_EVAL']]
        if not all(self.class_list):
            raise TrackEvalException('Attempted to evaluate an invalid class. Only classes [car, pedestrian] are valid.')
        #self.class_name_to_class_id = {'car': 1, 'van': 2, 'truck': 3, 'pedestrian': 4, 'person': 5,  # person sitting
        #                               'cyclist': 6, 'tram': 7, 'misc': 8, 'dontcare': 9, 'car_2': 1}
        self.class_name_to_class_id = {'dynamic':1}

        # Get sequences to eval and check gt files exist
        self.seq_list = []
        self.seq_lengths = {}

        # todo gt일 경우 이게 필요없게 수정.
        seqmap_name = 'evaluate_tracking.seqmap.' + self.config['SPLIT_TO_EVAL']
        seqmap_file = None
        if self.output_fol:
            seqmap_file = os.path.join(self.output_fol, seqmap_name) # gt의 seqmap을 봐야한다는게 웃기다.
            if not os.path.isfile(seqmap_file):
                raise TrackEvalException('no seqmap found: ' + os.path.basename(seqmap_file))

        if not seqmap_file: # i.e., self.output_fol == None
            ldir = os.path.join(self.gt_fol, 'label_02')
            lfiles = os.listdir(ldir)
            lfiles.sort()
            for lfile in lfiles:
                seq = os.path.splitext(lfile)[0]
                self.seq_list.append(seq)
                with open(os.path.join(ldir,lfile), 'r') as file:
                    lines = file.readlines()
                    last_line = lines[-1].strip()  # Remove any leading or trailing whitespaces
                    last_t = last_line.split(' ')[0]
                    self.seq_lengths[seq] = int(last_t)+1
        else:
            with open(seqmap_file) as fp:
                dialect = csv.Sniffer().sniff(fp.read(1024))
                fp.seek(0)
                reader = csv.reader(fp, dialect)
                for row in reader:
                    if len(row) >= 4:
                        seq = row[0]
                        self.seq_list.append(seq)
                        self.seq_lengths[seq] = int(row[3])
                        curr_file = os.path.join(self.gt_fol, 'label_02', seq + '.txt')
                        if not os.path.isfile(curr_file):
                            raise TrackEvalException('GT file not found: ' + os.path.basename(curr_file))

        # Get trackers to eval
        if self.config['TRACKERS_TO_EVAL'] is None:
            self.tracker_list = os.listdir(self.tracker_fol)
        else:
            self.tracker_list = self.config['TRACKERS_TO_EVAL']

        if self.config['TRACKER_DISPLAY_NAMES'] is None:
            self.tracker_to_disp = dict(zip(self.tracker_list, self.tracker_list))
        elif (self.config['TRACKERS_TO_EVAL'] is not None) and (
                len(self.config['TRACKER_DISPLAY_NAMES']) == len(self.tracker_list)):
            self.tracker_to_disp = dict(zip(self.tracker_list, self.config['TRACKER_DISPLAY_NAMES']))
        else:
            raise TrackEvalException('List of tracker files and tracker display names do not match.')

        if self.output_fol:
            for tracker in self.tracker_list:
                for seq in self.seq_list:
                    curr_file = os.path.join(self.tracker_fol, tracker, self.tracker_sub_fol, seq + '.txt')
                    if not os.path.isfile(curr_file):
                        raise TrackEvalException(
                            'Tracker file not found: ' + tracker + '/' + self.tracker_sub_fol + '/' + os.path.basename(
                                curr_file))

        return
