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
#from evaluate_rpe import evaluate_trajectory, transform44
import evaluate_rpe as rpe

from tabulate import tabulate


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
    interval = np.median([s-t for s,t in zip(stamps[1:],stamps[:-1])])
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
        f_instances = open(osp.join(self.trarget_dir, "instances.txt"), 'r')
        ins_reader = csv.reader(f_instances, delimiter=' ')
        eid2qth = {}
        for pth, qth in ins_reader:
            eid2qth[int(pth)] = int(qth)

        csv_file = open(osp.join(self.trarget_dir, "keypoints.txt"), 'r')
        csv_reader = csv.reader(csv_file, delimiter=' ')
        font_face, font_scale,font_thickness = cv2.FONT_HERSHEY_SIMPLEX, .5, 1
        N = len(self.est_ins_files)

        nkptTP, nkptTN, nkptFP, nkptFN = 0, 0, 0, 0
        TP,TN,FP,FN = [], [], [], []

        row = csv_reader.next()
        for n in range(N):
            #if n < 30: # TODO
            #    continue
            #if n > 32:
            #    break
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

            for eid in eids: # mask는 dynamic 판정이 뒤늦게 일어나므로, LBA widnow가 반영함에도, FN로 판정되는 케이스가 있음. 
                if eid > 0 and eid in eid2qth: # mappoint가 없는 instance(eid) 는 eid2qth에 저장안됨.
                    eid2ondynamic[eid] = eid2qth[eid] != 0
            #est_cp_locations, est_cp_distances = GetMarkerCenters(est_ins)
            #for eid, xy in est_cp_locations.items():
            #    eid2ondynamic[eid] = est_dmask[xy[1],xy[0]] > 0
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
                if not eid in eid2qth: # mappoint가 없는 instance(eid) 는 eid2qth에 저장안됨.
                    continue
                est_on_dynamic = eid2ondynamic[eid]

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
            if n % 100 == 0:
                print( "%d/%d"%(n, N) )
            #if not hasattr(self, 'dataset'):
            #    self.dataset = KittiTrackingDataset(self.dataset_dir,seq)
            #rgb = self.dataset[n][1]
            #dst = np.zeros(rgb.shape, rgb.dtype)
            #cases = {"TP":tps, "FP":fps, "FN":fns}
            #for case, infos in cases.items():
            #    for _, gid, _ in infos:
            #        cp = gt_cp_locations[gid]
            #        (txt_width, txt_height), _ = cv2.getTextSize(case,font_face, font_scale,font_thickness)
            #        if case == "TP":
            #            dst[gt_ins==gid,1] = 255
            #        else:
            #            dst[gt_ins==gid,2] = 255
            #        cv2.rectangle(dst, (cp[0], cp[1]-txt_height), (cp[0]+txt_width,cp[1]), (255,255,255), -1)
            #        cv2.putText(dst, case, cp, font_face,font_scale,(0,0,0),font_thickness)
            ##cv2.imshow("rgb", rgb)
            ##cv2.imshow("est_ins", GetColoredLabel(est_ins))
            ##cv2.imshow("gt_dmask", gt_dmask)
            #cv2.imshow("est_dmask", est_dmask)
            #cv2.imshow("dst", dst)
            #if ord('q') == cv2.waitKey():
            #    break
        table_data = (int(self.seq), "%d/%d"%(nkptTP+nkptTN,nkptFP+nkptFN), nkptTP, nkptTN, nkptFP, nkptFN)
        headers = ["Seq", "T/F", "TP", "TN", "FP", "FN"]
        return table_data, headers

    def eval_trj(self, ax=None):
        scale = 1.0
        offset = 0.
        max_difference = 0.02
        max_pairs = 10000
        fixed_delta = True
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
        first_xyz = np.matrix([[float(value) for value in first_list[a][0:3]] for a,b in matches]).transpose()
        second_xyz = np.matrix([[float(value)*float(scale) for value in second_list[b][0:3]] for a,b in matches]).transpose()
        rot,trans,trans_error = align(second_xyz,first_xyz)
        second_xyz_aligned = rot * second_xyz + trans
        first_stamps = first_list.keys()
        first_stamps.sort()
        first_xyz_full = np.matrix([[float(value) for value in first_list[b][0:3]] for b in first_stamps]).transpose()
        second_stamps = second_list.keys()
        second_stamps.sort()
        second_xyz_full = np.matrix([[float(value)*float(scale) for value in second_list[b][0:3]] for b in second_stamps]).transpose()
        second_xyz_full_aligned = rot * second_xyz_full + trans

        ate_rmse   = np.sqrt(np.dot(trans_error,trans_error) / len(trans_error))
        ate_median = np.median(trans_error)

        # convert to 4by4 trajectory
        traj_gt = dict([(k, rpe.transform44([k]+v)) for k,v in first_list.items()])
        traj_est = dict([(k,rpe.transform44([k]+v)) for k,v in second_list.items()])
        result = rpe.evaluate_trajectory(traj_gt, traj_est, max_pairs, fixed_delta, 1., 'm', offset, scale)
        trans_error = numpy.array(result)[:,4]
        rot_error = numpy.array(result)[:,5]
        rpe_meter_trans_rmse   = np.sqrt(np.dot(trans_error,trans_error) / len(trans_error))
        rpe_meter_trans_median = np.median(trans_error)
        rpe_meter_rot_rmse     = np.sqrt(np.dot(rot_error,rot_error) / len(rot_error)) * 180.0 / numpy.pi # [deg]
        rpe_meter_rot_median   = np.median(rot_error)

        headers = ["Seq", "ATE,RMSE [m]","ATE,median [m]", "RPE,RMSE [m/m]", "RPE,median[m/m]", "RPE,RMSE [deg/m]", "RPE,median[deg/m]" ]
        table_data = (int(self.seq), ate_rmse,ate_median,
                      rpe_meter_trans_rmse, rpe_meter_trans_median,
                      rpe_meter_rot_rmse, rpe_meter_rot_median )
        if ax is not None:
            plot_xz_traj(ax,first_stamps,first_xyz_full.transpose().A,'-',"black","ground truth")
            plot_xz_traj(ax,second_stamps,second_xyz_full_aligned.transpose().A,'-',"blue","estimated")
            ax.set_xlabel('x [m]')
            ax.set_ylabel('z [m]')
            ax.axis('equal')
        return table_data, headers 

def batch_evaluation(output_dir):
    dataset_path ="./kitti_tracking_dataset"
    targets = listdir(output_dir)
    targets = [x for x in targets if osp.isdir( osp.join(output_dir, x) )]
    targets = sorted(targets, key=lambda x: int(x.split('_')[1]) )
    fig = plt.figure(figsize=(8,12),dpi=100)
    plt.tight_layout()
    fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.4, hspace=1.)
    N, M = 4,6
    mask_datas, trj_datas = [], []
    for i,target in enumerate(targets):
        print("parsing for %s (%d/%d)..." % (target, i, len(targets)) )
        try:
            ax = fig.add_subplot(N,M,i+1)
        except:
            import pdb; pdb.set_trace()
        seq = int( target.split('_')[1] )
        ax.set_title('Seq %d'%seq)
        e = Evaluator(dataset_path, output_dir, target)
        mask_data, mask_header = e.eval_mask()
        mask_datas.append(mask_data)
        trj_data, trj_header = e.eval_trj(ax=ax)
        trj_datas.append(trj_data)

        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        ax.axis('equal')
        ax.set_aspect('equal', 'box')
        ax.figure.savefig(osp.join(output_dir,'trj_%d.png'%seq), dpi=100, bbox_inches=extent.expanded(1.6,1.6),)
    fig.savefig(osp.join(output_dir,'trajectories.png'),dpi=100)
    mask_table = tabulate(mask_datas, headers=mask_header, tablefmt="pretty")
    print(mask_table)
    trj_table = tabulate(trj_datas, headers=trj_header, tablefmt="pretty")
    print(trj_table)
    with open(osp.join(output_dir,"eval.txt"), "w") as file:
        file.write(mask_table)
        file.write(trj_table)
    with open(osp.join(output_dir,'eval.pkl'), 'wb') as f:
        pickle.dump({'mask_table':mask_table, 'trj_table':trj_table}, f)
    print("Done")
    #plt.show(block=True)
    return

def each_evaluation(target='training_0003'):
    dataset_path ="./kitti_tracking_dataset"
    output_dir = './output'
    e = Evaluator(dataset_path, output_dir, target)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    trj_data, trj_header = e.eval_trj(ax=ax)
    trj_table = tabulate([trj_data], headers=trj_header, tablefmt="pretty")
    print(trj_table)
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    ax.axis('equal')
    ax.set_aspect('equal', 'box')
    seq = int( target.split('_')[1] )
    fig.savefig(osp.join(output_dir,'trj_%d.png'%seq), dpi=100)
    mask_data, mask_header = e.eval_mask()
    mask_table = tabulate([mask_data], headers=mask_header, tablefmt="pretty")
    print(mask_table)
    plt.show(block=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output_dir', help='training_00xx or test_00xx or batch')
    args = parser.parse_args()
    if hasattr(args, 'output_dir'):
        batch_evaluation(args.output_dir)
    else:
        each_evaluation('output')
