#!/usr/bin/python2
#-*- coding:utf-8 -*-

import matplotlib as mpl
import matplotlib.pyplot as plt
from os import path as osp
from os import listdir
import pickle
import cv2
import numpy as np
import fnmatch

import sys
sys.path.append('kitti_tracking_tools')
from dataset.kitti_dataset import KittiTrackingDataset

from util import *
import csv

sys.path.append('tum_rgbd_benchmark_tools')
from converter import *
from associate import *
from evaluate_ate import align
from evaluate_rpe import evaluate_trajectory, transform44

def plot_xz_traj(ax,stamps,traj,style,color,label):
    """
    Plot a trajectory using matplotlib. 
    
    Input:
    ax -- the plot
    stamps -- time stamps (1xn)
    traj -- trajectory (3xn)
    style -- line style
    color -- line color
    label -- plot legend
    
    """
    stamps.sort()
    interval = numpy.median([s-t for s,t in zip(stamps[1:],stamps[:-1])])
    x = []
    z = []
    last = stamps[0]
    for i in range(len(stamps)):
        if stamps[i]-last < 2*interval:
            x.append(traj[i][0])
            z.append(traj[i][2])
        elif len(x)>0:
            ax.plot(x,y,style,color=color,label=label)
            label=""
            x=[]
            z=[]
        last= stamps[i]
    if len(x)>0:
        ax.plot(x,z,style,color=color,label=label)


class Evaluator:
    def __init__(self, dataset_path, output_dir, target):
        self.output_dir = output_dir
        self.trarget_dir = osp.join(output_dir, target)
        self.est_mask_dir = osp.join(self.trarget_dir, 'mask')
        assert osp.exists(self.est_mask_dir)

        est_mask_files = listdir( self.est_mask_dir )
        ins_files = fnmatch.filter(est_mask_files, 'ins*.raw')
        dmask_files = fnmatch.filter(est_mask_files, 'dynamic*.png')
        self.est_ins_files = sorted(ins_files, key=lambda file_name: int(file_name[3:-4]))
        self.est_dmask_files = sorted(dmask_files, key=lambda file_name: int(file_name[7:-4]))

        dataset_type, seq = target.split('_')
        self.dataset_dir = osp.join(dataset_path, dataset_type)
        self.gt_mask_dir = osp.join(self.dataset_dir, 'mask_02', seq)
        self.dataset_type = dataset_type
        self.seq = seq 

    def eval_mask(self):
        csv_file = open(osp.join(self.trarget_dir, "keypoints.txt"), 'r')
        csv_reader = csv.reader(csv_file, delimiter=' ')
        #print(csv_reader.next())
        #import pdb; pdb.set_trace()
        font_face, font_scale,font_thickness = cv2.FONT_HERSHEY_SIMPLEX, .5, 1
        N = len(self.est_ins_files)

        nkptTP, nkptTN, nkptFP, nkptFN = 0, 0, 0, 0
        TP,TN,FP,FN = [], [], [], []

        row = csv_reader.next()
        for n in range(N):
            if n < 30:
                continue
            est_ins_fn   = osp.join(self.est_mask_dir,self.est_ins_files[n])
            est_dmask_fn = osp.join(self.est_mask_dir,self.est_dmask_files[n])
            est_dmask = cv2.imread(est_dmask_fn, cv2.IMREAD_GRAYSCALE)
            gt_ins_fn   = osp.join(self.gt_mask_dir, 'ins%06d.raw'%n)
            gt_dmask_fn = osp.join(self.gt_mask_dir, 'dynamic%06d.png'%n)
            gt_dmask = cv2.imread(gt_dmask_fn, cv2.IMREAD_GRAYSCALE)
            gt_dmask[gt_dmask>0] = 255
            est_ins = ReadRawImage(est_ins_fn).copy()
            est_ins[est_ins<0] = 0
            gt_ins = ReadRawImage(gt_ins_fn)
            gt_est = np.vstack((gt_ins.reshape(-1,), est_ins.reshape(-1,))) # 2 x width*height
            gt_est_ids, cnts = np.unique(gt_est, return_counts=True, axis=1)
            gids, g_areas = np.unique(gt_ins,return_counts=True)
            eids, e_areas = np.unique(est_ins,return_counts=True)
            g2area, e2area = {}, {}
            gid2best_est = {}
            for id, area in zip(gids, g_areas):
                if id <= 0:
                    continue
                g2area[id] = area
                gid2best_est[id] = (-1, 0.) # id, area
            for id, area in zip(eids, e_areas):
                e2area[id] = area
            for (gid, eid), intersec in zip(gt_est_ids.T, cnts):
                if gid < 1:
                    continue
                if eid < 1:
                    continue
                garea = g2area[gid]
                earea = e2area[eid]
                iou = float(intersec) / float(garea+earea-intersec)
                if iou > gid2best_est[gid][1]:
                    gid2best_est[gid] = (eid, iou)

            # gt instance 별 dynamic 여부 확인
            gt_cp_locations, gt_cp_distances = GetMarkerCenters(gt_ins)
            gid2ondynamic, eid2ondynamic = {}, {-1:False}
            for gid, xy in gt_cp_locations.items():
                gid2ondynamic[gid] = gt_dmask[xy[1],xy[0]] > 0
            est_cp_locations, est_cp_distances = GetMarkerCenters(est_ins)
            for eid, xy in est_cp_locations.items():
                eid2ondynamic[eid] = est_dmask[xy[1],xy[0]] > 0
            """
            * confidence가 아니라서 mAP를 구하기 힘들다. minIoU로 TP,FP,FN만 구하자.
            * keypoints의 ins_id 를 기준으로, 
            """
            # Detection 여부 판정.
            tps, tns, fps, fns = [], [], [], []
            for gid, gt_on_dynamic in gid2ondynamic.items():
                eid, iou = gid2best_est[gid]
                info = (n, gid, eid )
                true_segmentation = iou > .5
                try:
                    est_on_dynamic = eid2ondynamic[eid]
                except:
                    import pdb; pdb.set_trace()

                if gt_on_dynamic == est_on_dynamic:
                    if gt_on_dynamic:
                        tps.append(info)
                    else:
                        tns.append(info)
                else:
                    if gt_on_dynamic:
                        fns.append(info)
                    else:
                        fps.append(info)

            keypoints = []
            while row is not None:
                frame_id = int(row[0])
                if frame_id < n:
                    row = csv_reader.next()
                    continue
                if frame_id > n:
                    break
                val1 = [int(v) for v in row[1:-2]]
                val2 = [float(v) for v in row[-2:]]
                val2 = [int(v) for v in val2]
                keypoints.append(val1+val2)
                try:
                    row = csv_reader.next()
                except:
                    break

            # kpt가 dynamic 판정인지 여부 / dynamic인지 여부.
            nkpt_tp, nkpt_tn, nkpt_fp, nkpt_fn = 0, 0, 0, 0
            for kpt_id, ins_id, has_mp, x,y in keypoints:
                if not has_mp:
                    continue
                est_on_dynamic = est_dmask[y,x]
                gt_on_dynamic  =  gt_dmask[y,x]
                if gt_on_dynamic == est_on_dynamic:
                    if gt_on_dynamic:
                        nkpt_tp +=1
                    else:
                        nkpt_tn +=1
                else:
                    if gt_on_dynamic:
                        nkpt_fn += 1
                    else:
                        nkpt_fp += 1

            nkptTP += nkpt_tp; nkptTN += nkpt_tn; nkptFP += nkpt_fp; nkptFN += nkpt_fn
            TP += tps; TN += tns; FP += fps; FN += fns;
            if n % 10 == 0:
                print( "%d/%d"%(n, N) )
            verbose = False
            if not verbose:
                continue
            if not hasattr(self, 'dataset'):
                self.dataset = KittiTrackingDataset(self.dataset_dir,seq)
            rgb = self.dataset[n][1]
            dst = np.zeros(rgb.shape, rgb.dtype)
            cases = {"TP":tps, "FP":fps, "FN":fns}
            for case, infos in cases.items():
                for _, gid, _ in infos:
                    cp = gt_cp_locations[gid]
                    (txt_width, txt_height), _ = cv2.getTextSize(case,font_face, font_scale,font_thickness)
                    if case == "TP":
                        dst[gt_ins==gid,1] = 255
                    else:
                        dst[gt_ins==gid,2] = 255
                    cv2.rectangle(dst, (cp[0], cp[1]-txt_height), (cp[0]+txt_width,cp[1]), (255,255,255), -1)
                    cv2.putText(dst, case, cp, font_face,font_scale,(0,0,0),font_thickness)
            #cv2.imshow("rgb", rgb)
            #cv2.imshow("est_ins", GetColoredLabel(est_ins))
            #cv2.imshow("gt_dmask", gt_dmask)
            cv2.imshow("est_dmask", est_dmask)
            cv2.imshow("dst", dst)
            if ord('q') == cv2.waitKey():
                break
        print("seq %s/%s"% (self.dataset_type,self.seq) )
        print( "kpt> T/F=%d/%d, nTP=%d, nTN=%d, nFP=%d, nFN=%d"  %(nkptTP+nkptTN, nkptFP+nkptFN,
                                                                  nkptTP, nkptTN, nkptFP, nkptFN) )
        print("ins> T/F=%d/%d, nTP=%d, nTN=%d, nFP=%d, nFN=%d"  %(len(TP)+len(TN), len(FP)+len(FN),
                                                                  len(TP), len(TN), len(FP), len(FN)) )
        return

    def eval_trj(self):
        scale = 1.0
        offset = 0.
        max_difference = 0.02
        max_pairs = 10000
        fixed_delta = False
        delta = 1.0
        delta_unit = 'm' # \'s\' for seconds, \'m\' for meters, \'rad\' for radians, \'f\' for frames
        '''
        TODO 
        '''
        fn_calib = osp.join(self.dataset_dir, 'calib', '%s.txt'%self.seq)
        fn_oxts = osp.join(self.dataset_dir, 'oxts', '%s.txt'%self.seq)
        first_list = read_kittiraw_oxts(fn_calib,fn_oxts)

        est_trj_fn = osp.join(self.trarget_dir, 'trj.txt')
        second_list = read_kittiodom_poses(est_trj_fn)
        matches = associate(first_list, second_list,float(offset),float(max_difference))    
        first_xyz = numpy.matrix([[float(value) for value in first_list[a][0:3]] for a,b in matches]).transpose()
        second_xyz = numpy.matrix([[float(value)*float(scale) for value in second_list[b][0:3]] for a,b in matches]).transpose()
        rot,trans,trans_error = align(second_xyz,first_xyz)
        second_xyz_aligned = rot * second_xyz + trans
        first_stamps = first_list.keys()
        first_stamps.sort()
        first_xyz_full = numpy.matrix([[float(value) for value in first_list[b][0:3]] for b in first_stamps]).transpose()
        second_stamps = second_list.keys()
        second_stamps.sort()
        second_xyz_full = numpy.matrix([[float(value)*float(scale) for value in second_list[b][0:3]] for b in second_stamps]).transpose()
        second_xyz_full_aligned = rot * second_xyz_full + trans

        #print "compared_pose_pairs %d pairs"%(len(trans_error))
        print "absolute_translational_error.rmse %f m"%numpy.sqrt(numpy.dot(trans_error,trans_error) / len(trans_error))
        print "absolute_translational_error.mean %f m"%numpy.mean(trans_error)
        print "absolute_translational_error.median %f m"%numpy.median(trans_error)
        print "absolute_translational_error.std %f m"%numpy.std(trans_error)
        print "absolute_translational_error.min %f m"%numpy.min(trans_error)
        print "absolute_translational_error.max %f m"%numpy.max(trans_error)

        # convert to 4by4 trajectory
        traj_gt = dict([(k,transform44([k]+v)) for k,v in first_list.items()])
        traj_est = dict([(k,transform44([k]+v)) for k,v in second_list.items()])
        result = evaluate_trajectory(traj_gt, traj_est, max_pairs, fixed_delta, delta, delta_unit, offset, scale)
        stamps = numpy.array(result)[:,0]
        trans_error = numpy.array(result)[:,4]
        rot_error = numpy.array(result)[:,5]
        print("RPE per %.2f[%s] >" %  (delta,delta_unit) )
        print ("translational_error.rmse %f m"%numpy.sqrt(numpy.dot(trans_error,trans_error) / len(trans_error)))
        print ("translational_error.mean %f m"%numpy.mean(trans_error))
        print ("translational_error.median %f m"%numpy.median(trans_error))
        print ("translational_error.std %f m"%numpy.std(trans_error))
        print ("translational_error.min %f m"%numpy.min(trans_error))
        print ("translational_error.max %f m"%numpy.max(trans_error))
        print ("rotational_error.rmse %f deg"%(numpy.sqrt(numpy.dot(rot_error,rot_error) / len(rot_error)) * 180.0 / numpy.pi))
        print ("rotational_error.mean %f deg"%(numpy.mean(rot_error) * 180.0 / numpy.pi))
        print ("rotational_error.median %f deg"%numpy.median(rot_error))
        print ("rotational_error.std %f deg"%(numpy.std(rot_error) * 180.0 / numpy.pi))
        print ("rotational_error.min %f deg"%(numpy.min(rot_error) * 180.0 / numpy.pi))
        print ("rotational_error.max %f deg"%(numpy.max(rot_error) * 180.0 / numpy.pi))

        fig = plt.figure()
        ax = fig.add_subplot(111)
        plot_xz_traj(ax,first_stamps,first_xyz_full.transpose().A,'-',"black","ground truth")
        plot_xz_traj(ax,second_stamps,second_xyz_full_aligned.transpose().A,'-',"blue","estimated")
        ax.legend()
        ax.set_xlabel('x [m]')
        ax.set_ylabel('z [m]')
        ax.axis('equal') #ax.set_aspect('equal', 'box')
        plt.show(block=True)
        return

if __name__ == '__main__':
    dataset_path ="./kitti_tracking_dataset"
    output_dir = './output'
    target = 'training_0003'
    
    evaluator = Evaluator(dataset_path, output_dir, target)
    #evaluator.eval_mask()
    evaluator.eval_trj()


