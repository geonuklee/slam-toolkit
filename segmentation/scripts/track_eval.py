#!/usr/bin/python3'THRESHOLD':.2
#-*- coding:utf-8 -*-

import KittiTrackingTool as KTT
import cv2
import numpy as np
import trackeval  # noqa: E402
from multiprocessing import freeze_support
from os import path as osp

"""
* 주요 기능
    * KITTI tracking dataset의 parsing, visualization
        * https://github.com/hailanyi/3D-Detection-Tracking-Viewer 의 코드에서
        * oxts(gps/imu) parsing 추가.
    * 동적 물체의 mask, motioninstance_02 추가.
    * KITTI tracking datset과 동일한 sequence의 MOTS png label (KITTIMOTS /Annotations, instances) 지원.
        * Dataset homepage : https://www.vision.rwth-aachen.de/page/mots
    * TrackEval의 _BaseDataset 지원, MOTS evaluaiton 지원.

"""

def main():
    freeze_support()
    default_eval_config = trackeval.Evaluator.get_default_eval_config()
    default_eval_config['DISPLAY_LESS_PROGRESS'] = False
    default_dataset_config = KTT.KittiTrackingDataset.get_default_dataset_config()
    default_metrics_config = {'METRICS': ['CLEAR']}
    config = {**default_eval_config, **default_dataset_config, **default_metrics_config}  # Merge default configs

    eval_config = {k: v for k, v in config.items() if k in default_eval_config.keys()}
    dataset_config = {k: v for k, v in config.items() if k in default_dataset_config.keys()}
    metrics_config = {k: v for k, v in config.items() if k in default_metrics_config.keys()}
    evaluator = trackeval.Evaluator(eval_config)

    eval_threshold = .4


    """
    default_dataset_config['TRACKERS_FOLDER'] = data/gt/kitti/kitti_2d_box_train
    default_dataset_config['GT_FOLDER'] = data/gt/kitti/kitti_2d_box_train
    """
    dataset_config['GT_FOLDER'] = '/home/geo/dataset/kitti_tracking_dataset/%s' % dataset_config['SPLIT_TO_EVAL']

    dataset_config['TRACKER_SUB_FOLDER'] = ''
    dataset_config['TRACKERS_FOLDER'] = '/home/geo/ws/slam-toolkit/segmentation/output_batch_:0/trackevalform_%s' % dataset_config['SPLIT_TO_EVAL']
    dataset_config['TRACKERS_TO_EVAL'] = ['MySeg']
    dataset = KTT.KittiTrackingDataset(dataset_config)# , dataset_config['GT_FOLDER'], seq)

    do_visualization = False 
    if do_visualization:
        targets = ['gt', 'tracker']
        font, fs, ft = cv2.FONT_HERSHEY_PLAIN, 1.5, 1
        # TODO dataset.get_preprocessed_seq_data(raw_data, 'dynamic') 으로 matching visualization 필요.
        for seq in dataset.seq_list:
            raw_datas = {}
            raw_datas['gt'] = dataset._load_raw_file('', seq, is_gt=True)
            if 'tracker' in targets:
                raw_datas['tracker'] = dataset._load_raw_file('MySeg', seq, is_gt=False)

            matched_raw_datas = dataset.get_raw_seq_data('MySeg', seq) # _BaseDataset.get_raw_seq_data
            """ keys
            'tracker_confidences', 'tracker_classes', 'tracker_dets', 'num_timesteps', 'seq',
                'gt_crowd_ignore_regions', 'gt_extras', 'gt_ids', 'gt_classes', 'gt_dets', 
                'tracker_ids', 'gt_ids'  둘은 matching이 되서 연결되다고 전재. trackeval/metric/clear.py
                'similarity_scores' << IoU
            """

            for t in range(dataset.seq_lengths[seq]):
                src = cv2.imread(osp.join(dataset.lcolor_path, str(t).zfill(6)+'.png'))
                dst = cv2.vconcat( (src,src) )
                yoffset = src.shape[0]

                tracker_dets = matched_raw_datas['tracker_dets'][t]
                gt_dets      = matched_raw_datas['gt_dets'][t]
                similarities = matched_raw_datas['similarity_scores'][t]
                gt_ids       = matched_raw_datas['gt_ids'][t]
                tracker_ids  = matched_raw_datas['tracker_ids'][t]
                if tracker_dets.shape[0] < 1:
                    continue
                j_matches = np.argmax(similarities, axis=1) 
                s_matches = np.max(similarities, axis=1)

                for i, (j, iou) in enumerate(zip(j_matches,s_matches)):
                    bbox0 = gt_dets[i].astype(np.int)
                    bbox1 = tracker_dets[j].astype(np.int)
                    cp0 = ( int((bbox0[0]+bbox0[2])/2), bbox0[3] )
                    cp1 = ( int((bbox1[0]+bbox1[2])/2), yoffset+bbox1[1] )
                    if iou > eval_threshold:
                        color = (0,255,0)
                    elif iou > .1:
                        color = (100,100,100)
                    else:
                        continue
                    cv2.line(dst, cp0, cp1, color, 2)

                for j, det in enumerate(tracker_dets):
                    cls = matched_raw_datas['tracker_classes'][t][j]
                    _id = tracker_ids[j]
                    bbox = det.astype(np.int).tolist()
                    bbox[1] += yoffset;
                    bbox[3] += yoffset;
                    bbox = tuple(bbox)

                    matched = False
                    if j in j_matches:
                        i = np.where(j_matches==j)[0][0] 
                        if s_matches[i]  > eval_threshold :
                            matched = True
                    if matched:
                        color = (0,255,0)
                    else:
                        color = (0,0, 255)
                    cv2.rectangle(dst, bbox[:2], bbox[2:], color, 2)

                    msg = "%s"%_id
                    size, _ = cv2.getTextSize(msg,font,fs,ft)
                    cv2.rectangle(dst, (bbox[0]-2, bbox[1]+2), (bbox[0]+size[0]+2, bbox[1]-size[1]-2), (255,255,255),-1)
                    cv2.putText(dst, msg, (bbox[0], bbox[1]),font,fs,(0,0,0),ft)

                for i, det in enumerate(gt_dets):
                    cls = matched_raw_datas['gt_classes'][t][i]
                    _id = gt_ids[i]
                    bbox = tuple(det.astype(np.int))

                    if s_matches[i] > eval_threshold :
                        color = (0,255,0)
                    else:
                        color = (0, 0, 255)
                    cv2.rectangle(dst, bbox[:2], bbox[2:], color, 2)

                    msg = "%s %.2f"%(_id, s_matches[i] )
                    size, _ = cv2.getTextSize(msg,font,fs,ft)
                    cv2.rectangle(dst, (bbox[0]-2, bbox[1]+2), (bbox[0]+size[0]+2, bbox[1]-size[1]-2), (255,255,255),-1)
                    cv2.putText(dst, msg, (bbox[0], bbox[1]),font,fs,(0,0,0),ft)
                cv2.imshow("dst", dst)
                c = cv2.waitKey()
                if c==ord('q'):
                    break
            # for t
        # for seq

    do_evaluation = True
    if do_evaluation:
        dataset_list = [dataset]
        metrics_list = []
        for metric in [trackeval.metrics.HOTA, trackeval.metrics.CLEAR, trackeval.metrics.Identity]:
            if metric.get_name() in metrics_config['METRICS']:
                metrics_list.append(metric(config={'THRESHOLD': eval_threshold}))
        if len(metrics_list) == 0:
            raise Exception('No metrics selected for evaluation')
        evaluator.evaluate(dataset_list, metrics_list)

    return



if __name__ == '__main__':
    '''
    python3 setup.py install --user
    python3 scripts/track_eval.py
    '''
    main()

