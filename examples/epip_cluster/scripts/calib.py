#!/usr/bin/python

import cv2
import numpy as np

import pickle
from os import path as osp
from os import listdir

# Parameters
# Denote! : Pattern size must be (number of horizontal, number of vertical)
# If the order is wrong, then wrong chess board detection cause calibration failure
# About Calibration
pattern_size = (8,6) # << Important!
pattern_length = 0.108
chessboard_flag = cv2.CALIB_CB_FAST_CHECK;

(major, minor, _) = cv2.__version__.split(".")
calibration_term_crit = (cv2.TERM_CRITERIA_MAX_ITER|cv2.TERM_CRITERIA_EPS, 30, 1e-6) # default 1e-6


init_fx = 400.
min_median = 100 # pixel

# Etc, visualization
max_width_of_visualize_window = 1000
horizontal_line_offset = 30

class StereoDataset(object):
    def __init__(self, seq_path):
        for i, sub in enumerate(['image0', 'image1']):
            sub_dir = osp.join(seq_path, sub)
            sub_files = listdir(sub_dir)
            l = []
            for n in range(len(sub_files) ):
                fn = osp.join(sub_dir,'%d.png'%n)
                l.append(fn)
            if i == 0:
                self.left_files = l
            else:
                self.right_files = l
        assert len(self.left_files) == len(self.right_files)
        self.i = 0

    def __iter__(self):
        return self

    def next(self):
        if self.i < len(self.left_files):
            l_im = cv2.imread(self.left_files[self.i])
            r_im = cv2.imread(self.right_files[self.i])
            self.i += 1
        else:
            raise StopIteration()
        return l_im, r_im

    def rewind(self):
        self.i =0

    def __del__(self):
        pass

def resize_image(max_width, src):
    if src.shape[1] <= max_width:
        return src
    w = max_width
    h = float(src.shape[0])/float(src.shape[1])*w
    h = int(h)
    dst = cv2.resize(src, (w,h))
    return dst

def calibration(stereo_dataset):
    # Save chessboard detection to file, to avoid chessboard detection from same dataset at second time.
    fn_pick = 'calib.pick'

    obj_points = np.zeros((np.prod(pattern_size), 3), np.float32)
    obj_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    obj_points *= pattern_length

    cb_detection = {}

    # Check pickle
    if osp.isfile(fn_pick):
        print 'load pickle %s' % fn_pick
        fp = open(fn_pick,"rb")
        cb_detection = pickle.load(fp)
        fp.close()
        print 'done'
    else:
        cb_detection["group_l_corners"] = []
        cb_detection["group_r_corners"] = []
        previous_corrner = []

        for l_im, r_im in stereo_dataset:
            cb_detection["image_size"] = (l_im.shape[1], l_im.shape[0])
            l_found, l_corners = cv2.findChessboardCorners(l_im, pattern_size,flags=chessboard_flag)
            r_found, r_corners = cv2.findChessboardCorners(r_im, pattern_size,flags=chessboard_flag)

            if not l_found or not r_found:
                continue

            if len(previous_corrner) > 0:
                dists = []
                for i, pt in enumerate(l_corners):
                    pt0 = previous_corrner[i]
                    dist = np.linalg.norm(pt - pt0)
                    dists.append(dist)
                median = np.median(dists)
                if median < min_median:
                    continue
            previous_corrner = l_corners
            cb_detection["group_l_corners"].append(l_corners)
            cb_detection["group_r_corners"].append(r_corners)
            l_dst = cv2.drawChessboardCorners(l_im, pattern_size, l_corners, True)
            r_dst = cv2.drawChessboardCorners(r_im, pattern_size, r_corners, True)
            dst = np.hstack((l_dst,r_dst))
            dst = resize_image(max_width_of_visualize_window, dst)
            cv2.imshow("chess", dst)
            c = cv2.waitKey(1)
            if c == ord('q'):
                break
        print 'save pickle %s' % fn_pick
        print 'len = %d' % len(cb_detection["group_l_corners"])
        fp = open(fn_pick,"wb")
        pickle.dump(cb_detection, fp)
        fp.close()
        print 'done'

    cb_detection["group_obj_points"] = []
    for i in xrange(len(cb_detection["group_l_corners"])):
        cb_detection["group_obj_points"].append(obj_points)

    init_K = np.array((init_fx, 0., cb_detection["image_size"][0]/2.,
                       0., init_fx, cb_detection["image_size"][1]/2.,
                       0., 0, 1.)).reshape((3,3))
    init_d = np.array((0.,0.,0.,0.))

    metric = {}

    _, metric['K1'], metric['d1'], _, _\
            = cv2.calibrateCamera(cb_detection["group_obj_points"][:],
                    cb_detection["group_l_corners"][:],
                    cb_detection["image_size"],
                    init_K.copy(), init_d.copy() ) # Warnning! To prevent overwriting K2 on K1

    _, metric['K2'], metric['d2'], _, _\
            = cv2.calibrateCamera(cb_detection["group_obj_points"][:],
                    cb_detection["group_r_corners"][:],
                    cb_detection["image_size"],
                    init_K.copy(), init_d.copy() )

    calibration_flag = 0
    calibration_flag |= cv2.CALIB_FIX_INTRINSIC

    (metric['retval'], metric['K1'], metric['d1'], metric['K2'], metric['d2'],
     metric['R'], metric['t'], metric['E'],
     metric['F']) = cv2.stereoCalibrate(
             cb_detection["group_obj_points"][:],
             cb_detection["group_l_corners"][:],
             cb_detection["group_r_corners"][:],
             metric['K1'],
             metric['d1'],
             metric['K2'],
             metric['d2'],
             imageSize=cb_detection["image_size"],
             flags=calibration_flag,
             criteria=calibration_term_crit
             )
    metric["image_size"] = cb_detection["image_size"]

    print "K1 =\n",  metric['K1']
    print "d1 = ",   metric['d1'].T
    print "K2 =\n ", metric['K2']
    print "d2 = ",   metric['d2'].T
    print "R =\n",   metric['R']
    print "t = ",    metric['t'].T

    return metric


def show_rectification(stereo_dataset, metric):
    R1,R2,P1,P2,Q,roi1,roi2 = cv2.stereoRectify(metric['K1'], metric['d1'],metric['K2'], metric['d2'], metric['image_size'], metric['R'], metric['t'], flags=cv2.CALIB_ZERO_DISPARITY)
    print "R1 = \n" , R1
    print "R2 = \n" , R2
    print "P1 = \n" , P1
    print "P2 = \n" , P2
    print "roi1 = \n" , roi1
    print "roi2 = \n" , roi2

    new_img_size=metric['image_size']
    newK = np.matrix(P2[:,:3])

    mp1x, mp1y = cv2.initUndistortRectifyMap(metric['K1'], metric['d1'], R1, newK, new_img_size, cv2.CV_32FC1)
    mp2x, mp2y = cv2.initUndistortRectifyMap(metric['K2'], metric['d2'], R2, newK, new_img_size, cv2.CV_32FC1)

    for l_im, r_im in stereo_dataset:
        l_rec = cv2.remap(l_im, mp1x, mp1y, cv2.INTER_LINEAR)
        r_rec = cv2.remap(r_im, mp2x, mp2y, cv2.INTER_LINEAR)
        im_z = np.zeros(l_rec.shape, l_rec.dtype)

        l_rec_dst = l_rec.copy()
        r_rec_dst = r_rec.copy()
        cv2.rectangle(l_rec_dst, roi1[:2], (roi1[0]+roi1[2], roi1[1]+roi1[3]), (0,255,0),2)
        cv2.rectangle(r_rec_dst, roi2[:2], (roi2[0]+roi2[2], roi2[1]+roi2[3]), (0,255,0),2)

        dst0 = np.hstack((l_rec_dst, r_rec_dst))
        dst1 = np.hstack((r_rec, im_z))
        dst = np.vstack((dst0,dst1))
        #cv2.putText(dst, "i=%d"%i, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1., (0,0,255), 2)
        cv2.putText(dst, "green line for ROI", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1., (0,255,0), 2)
        cv2.putText(dst, "RecL", (10,100), cv2.FONT_HERSHEY_SIMPLEX, 1., (0,0,255), 2)
        cv2.putText(dst, "RecR", (l_rec.shape[1]+10,50), cv2.FONT_HERSHEY_SIMPLEX, 1., (0,0,255), 2)
        cv2.putText(dst, "RecR", (10,l_rec.shape[0]+50), cv2.FONT_HERSHEY_SIMPLEX, 1., (0,0,255), 2)
        dst = resize_image(max_width_of_visualize_window, dst)
        y = 0
        while y+horizontal_line_offset < dst.shape[0]/2:
            y+=horizontal_line_offset
            cv2.line(dst, (0,y), (dst.shape[1],y), (255,0,0))
        cv2.imshow("dst", dst)
        c = cv2.waitKey(0)
        if c == ord('q'):
            break

if __name__ == '__main__':
    seq_path = '/home/geo/dataset/stereo_dataset/2021-07-05/seq-2021-07-05-12:21:08-chessboard'
    stereo_dataset = StereoDataset(seq_path)
    metric = calibration(stereo_dataset)

    stereo_dataset = StereoDataset(seq_path)
    show_rectification(stereo_dataset, metric)
