# -*- coding: utf-8 -*-
import numpy as np
from os import path as osp
import cv2
from .dataset.kitti_dataset import *
from .viewer.color_map import generate_objects_color_map,generate_objects_colors,generate_scatter_colors
from scipy.spatial.transform import Rotation as rotation_util

"""
import sys; sys.path.append("/home/geo/ws/slam-toolkit/segmentation")
import kitti_tracking_tools as kt
kt.Parse("./kitti_tracking_dataset", "training", "0000")
"""

gui = False
if gui:
    import vedo
    import vtk

def box_op_velo_to_cam(cloud,vtc_mat):
    """
    description: convert Lidar 3D coordinates to 3D camera coordinates .
    input: (PointsNum,3)
    output: (PointsNum,3)
    """
    mat=np.ones(shape=(cloud.shape[0],4),dtype=np.float32)
    mat[:,0:3]=cloud[:,0:3]
    mat=np.mat(mat)
    normal=np.mat(vtc_mat)
    transformed_mat = normal * mat.T
    T=np.array(transformed_mat.T,dtype=np.float32)
    return T

def get_box_points(points, pose=None, point_num=200, show_box_heading=True):
    """
    box to points
    :param points: (7,),box
    :param pose:
    :return:
    """
    PI=np.pi
    import math
    point=np.zeros(shape=points.shape)
    point[:]=points[:]

    h,w,l = point[5],point[4],point[3]
    x,y,z = point[0],point[1],point[2]
    i=1
    label=1
    z_vector = np.arange(- h / 2, h / 2, h / point_num)[0:point_num]
    w_vector = np.arange(- w / 2, w / 2, w / point_num)[0:point_num]
    l_vector = np.arange(- l / 2, l / 2, l / point_num)[0:point_num]

    d_z_p = -np.sort(-np.arange(0, h / 2, h / (point_num*2))[0:point_num])
    d_z_n = np.arange( -h / 2,0, h / (point_num*2))[0:point_num]


    d_w_p = -np.sort(-np.arange(0, w / 2, w / (point_num*2))[0:point_num])
    d_w_n = np.arange(-w / 2,0,  w / (point_num*2))[0:point_num]

    d_l_p = np.arange(l / 2, l*(4/7) , (l*(4/7)-l / 2) / (point_num*2))[0:point_num]


    d1 = np.zeros(shape=(point_num, 4))
    d1[:, 0] = d_w_p
    d1[:, 1] = d_l_p
    d1[:, 2] = d_z_p
    d1[:, 3] = i

    d2 = np.zeros(shape=(point_num, 4))
    d2[:, 0] = d_w_n
    d2[:, 1] = d_l_p
    d2[:, 2] = d_z_p
    d2[:, 3] = i

    d3 = np.zeros(shape=(point_num, 4))
    d3[:, 0] = d_w_p
    d3[:, 1] = d_l_p
    d3[:, 2] = d_z_n
    d3[:, 3] = i

    d4 = np.zeros(shape=(point_num, 4))
    d4[:, 0] = d_w_n
    d4[:, 1] = d_l_p
    d4[:, 2] = d_z_n
    d4[:, 3] = i

    z1 = np.zeros(shape=(point_num, 4))
    z1[:, 0] = -w / 2
    z1[:, 1] = -l / 2
    z1[:, 2] = z_vector
    z1[:, 3] = i
    z2 = np.zeros(shape=(point_num, 4))
    z2[:, 0] = -w / 2
    z2[:, 1] =l / 2
    z2[:, 2] = z_vector
    z2[:, 3] = i
    z3 = np.zeros(shape=(point_num, 4))
    z3[:, 0] = w / 2
    z3[:, 1] = -l / 2
    z3[:, 2] = z_vector
    z3[:, 3] = i
    z4 = np.zeros(shape=(point_num, 4))
    z4[:, 0] = w / 2
    z4[:, 1] = l / 2
    z4[:, 2] = z_vector
    z4[:, 3] = i
    w1 = np.zeros(shape=(point_num, 4))
    w1[:, 0]=w_vector
    w1[:, 1]=-l / 2
    w1[:, 2]=-h / 2
    w1[:, 3] = i
    w2 = np.zeros(shape=(point_num, 4))
    w2[:, 0] = w_vector
    w2[:, 1] = -l/ 2
    w2[:, 2] = h / 2
    w2[:, 3] = i
    w3 = np.zeros(shape=(point_num, 4))
    w3[:, 0] = w_vector
    w3[:, 1] = l / 2
    w3[:, 2] = -h / 2
    w3[:, 3] = i
    w4 = np.zeros(shape=(point_num, 4))
    w4[:, 0] = w_vector
    w4[:, 1] =l / 2
    w4[:, 2] = h / 2
    w4[:, 3] = i
    l1 = np.zeros(shape=(point_num, 4))
    l1[:, 0] = -w / 2
    l1[:, 1] = l_vector
    l1[:, 2] = -h / 2
    l1[:, 3] = i
    l2 = np.zeros(shape=(point_num, 4))
    l2[:, 0] = -w / 2
    l2[:, 1] = l_vector
    l2[:, 2] = h / 2
    l2[:, 3] = i
    l3 = np.zeros(shape=(point_num, 4))
    l3[:, 0] = w / 2
    l3[:, 1] = l_vector
    l3[:, 2] = -h / 2
    l3[:, 3] = i
    l4 = np.zeros(shape=(point_num, 4))
    l4[:, 0] = w / 2
    l4[:, 1] = l_vector
    l4[:, 2] = h / 2
    l4[:, 3] = i

    if show_box_heading:
        point_mat = np.mat(np.concatenate((z1, z2, z3, z4, w1, w2, w3, w4, l1, l2, l3, l4, d1, d2, d3, d4)))
    else:
        point_mat = np.mat(np.concatenate((z1, z2, z3, z4, w1, w2, w3, w4, l1, l2, l3, l4)))

    angle=point[6]-PI/2

    if pose is None:
        convert_mat = np.mat([[math.cos(angle), -math.sin(angle), 0, x],
                              [math.sin(angle), math.cos(angle), 0, y],
                              [0, 0, 1, z],
                              [0, 0, 0, label]])

        transformed_mat = convert_mat * point_mat.T
    else:

        convert_mat = np.mat([[math.cos(angle), -math.sin(angle), 0, 0],
                              [math.sin(angle), math.cos(angle), 0, 0],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])
        transformed_mat = convert_mat * point_mat.T
        pose_mat = np.mat([[pose[0, 0], pose[0, 1], pose[0, 2], x],
                           [pose[1, 0], pose[1, 1], pose[1, 2], y],
                           [pose[2, 0], pose[2, 1], pose[2, 2], z],
                           [0, 0, 0, label]])
        transformed_mat = pose_mat * transformed_mat


    transformed_mat = np.array(transformed_mat.T,dtype=np.float32)

    return transformed_mat

def convert_box_type(boxes,input_box_type = 'Kitti'):
    """
    convert the box type to unified box type
    :param boxes: (array(N,7)), input boxes
    :param input_box_type: (str), input box type
    :return: new boxes with box type [x,y,z,l,w,h,yaw]
    """
    boxes = np.array(boxes)
    if len(boxes) == 0:
        return None
    assert  input_box_type in ["Kitti","OpenPCDet","Waymo"], 'unsupported input box type!'

    if input_box_type in ["OpenPCDet","Waymo"]:
        return boxes

    if input_box_type == "Kitti": #(h,w,l,x,y,z,yaw) -> (x,y,z,l,w,h,yaw)
        boxes = np.array(boxes)
        new_boxes = np.zeros(shape=boxes.shape)
        new_boxes[:,:]=boxes[:,:]
        new_boxes[:,0:3] = boxes[:,3:6]
        new_boxes[:, 3] = boxes[:, 2]
        new_boxes[:, 4] = boxes[:, 1]
        new_boxes[:, 5] = boxes[:, 0]
        new_boxes[:, 6] = (np.pi - boxes[:, 6]) + np.pi / 2
        new_boxes[:, 2] += boxes[:, 0] / 2
        return new_boxes

def get_mesh_boxes(boxes,colors="red",
                   mesh_alpha=0.4,
                   ids=None,
                   show_ids=False,
                   box_info=None,
                   show_box_info=False,
                   caption_size=(0.05,0.05)):
    """
    convert boxes array to vtk mesh boxes actors
    :param boxes: (array(N,7)), unified boxes array
    :param colors: (str or array(N,3)), boxes colors
    :param mesh_alpha: boxes transparency
    :param ids: list(N,), the ID of each box
    :param show_ids: (bool), show object ids in the 3D scene
    :param box_info: (list(N,)), a list of str, the infos of boxes to show
    :param show_box_info: (bool)，show object infos in the 3D Scene
    :return: (list(N,)), a list of vtk mesh boxes
    """
    vtk_boxes_list = []
    for i in range(len(boxes)):
        box = boxes[i]
        angle = box[6]

        new_angle = (angle / np.pi) * 180

        if type(colors) is str:
            this_c = colors
        else:
            this_c = colors[i]
        vtk_box = Box(pos=(0, 0, 0), height=box[5], width=box[4], length=box[3], c=this_c, alpha=mesh_alpha)
        vtk_box.rotateZ(new_angle)
        vtk_box.pos(box[0], box[1], box[2])

        info = ""
        if ids is not None and show_ids :
            info = "ID: "+str(ids[i])+'\n'
        if box_info is not None and show_box_info:
            info+=str(box_info[i])
        if info !='':
            vtk_box.caption(info,point=(box[0],
                            box[1]-box[4]/4, box[2]+box[5]/2),
                            size=caption_size,
                            alpha=1,c=this_c,
                            font="Calco",
                            justify='left')
            vtk_box._caption.SetBorder(False)
            vtk_box._caption.SetLeader(False)

        vtk_boxes_list.append(vtk_box)

    return vtk_boxes_list

class Parser:
    """
    Trajectory와 global 좌표상의 obj box를 표시하는 viewer
    default box type: "OpenPCDet", (x,y,z,l,w,h,yaw)
    """
    def __init__(self,dataset,box_type="Kitti",bg=(255, 255, 255)):
        self.objects_color_map = generate_objects_color_map('rainbow')
        self.box_type = box_type

        if gui:
            self.plt = vedo.Plotter(bg=bg)
        else:
            self.plt = None
        self.dataset = dataset
        poses = dataset.poses
        trj = []
        for pose in poses:
            trj.append( pose[:3] ) # T_w_c2's transloation
        self.trj = np.array(trj) # N by 3
        self.T0_for_id = {}
        self.set_dynamic_obj = set()

        if gui:
            self.all_trj_actor = vedo.Line(self.trj, lw=1,c=(.5,.5,.5))
            # Set the camera position to achieve the bird's-eye view
            self.reset_camera()
        self.i = 0

    def reset_camera(self):
        self.plt.camera.SetPosition(0, -100, 0)
        self.plt.camera.SetFocalPoint(0, 0, 0)
        self.plt.camera.SetViewUp(0, 0, 1)
        self.plt.reset_camera()
        self.plt.render()
        return

    def do(self):
        if self.i == self.trj.shape[0]-1:
            return False

        if self.plt is not None:
            self.plt.clear()
            if self.i > 0:
                curr_trj_actor = vedo.Line(self.trj[:self.i,:], lw=2,c=(1.,0.,0.))
            else:
                curr_trj_actor = None

        # animate your stuff here
        points, image, r_image, oxt, pose, labels, label_names = self.dataset[self.i]

        dst_color = image.copy()
        vis_instance = np.zeros_like(dst_color)
        dynamic_mask = np.zeros(image.shape[:2],np.uint8)


        box_actors = {}
        if label_names is not None:
            # classes = Car Van Truck Pedestrian Person_sitting Cyclist Tram Misc DontCare
            mask = label_names != "DontCare"

            labels = labels[mask]
            label_names = label_names[mask]

            # ids for..
            ids = labels[:,-1].astype(np.int32)
            boxes = convert_box_type(labels)

            T_w_c2 = np.eye(4)
            if hasattr(rotation_util, "as_matrix"):
                T_w_c2[:3,:3] = rotation_util.from_quat( pose[3:] ).as_matrix()
            else:
                T_w_c2[:3,:3] = rotation_util.from_quat( pose[3:] ).as_dcm()
            T_w_c2[:3,-1] = pose[:3]
            T_w_v = np.matmul(T_w_c2, self.dataset.T_c2_v)

            #sorted_indices = np.argsort(boxes[:, 0])[::-1] # x-axis forward on velodyne coordinate
            l = np.linalg.norm(boxes[:,:3],axis=1)
            sorted_indices = np.argsort(l)[::-1] # x-axis forward on velodyne coordinate

            boxes = boxes[sorted_indices]
            ids   = ids[sorted_indices]
            H,W,_ = image.shape
            colors = generate_objects_colors(ids,self.objects_color_map)
            for i in range(len(boxes)):
                # ref : https://github.com/utiasSTARS/pykitti/blob/master/pykitti/tracking.py#L222
                box, id = boxes[i], ids[i]
                this_c = colors[i]

                rad = box[6] # radian
                # Box는 velodyne 좌표에서 그려진거라, show2D는 vel_to_cam을 호출한다.
                T_v_b = np.eye(4)
                if hasattr(rotation_util, "as_matrix"):
                    T_v_b[:3,:3] = rotation_util.from_euler('z',rad).as_matrix()
                else:
                    T_v_b[:3,:3] = rotation_util.from_euler('z',rad).as_dcm()
                T_v_b[:3,-1] = box[:3]
                T_w_b = np.matmul(T_w_v, T_v_b)
                if id in self.T0_for_id:
                    t_err = T_w_b[:3,-1] - self.T0_for_id[id][:3,-1]
                    t_err = np.linalg.norm(t_err)
                else:
                    t_err = 0.
                    self.T0_for_id[id] = T_w_b
                if t_err > 2.:
                    self.set_dynamic_obj.add(id)
                is_dynamic = id in self.set_dynamic_obj
                if self.plt is not None:
                    vtk_box = vedo.Box(pos=(0,0,0), height=box[5], width=box[4], length=box[3], c=this_c,
                                       alpha=1. if is_dynamic else .2)
                    rot = rotation_util.from_matrix(T_w_b[:3,:3]).as_euler('xyz', degrees=True).astype(np.int32)
                    msg = "%s#%d:%s" % (label_names[i], id, 'dynamic' if is_dynamic else 'static' )
                    vtk_box.rotate_x(rot[0])
                    vtk_box.rotate_y(rot[1])
                    vtk_box.rotate_z(rot[2])
                    vtk_box.pos(T_w_b[:3,3])
                    vtk_box.caption(msg, point=T_w_b[:3,-1],
                                    alpha=1.,
                                    c='black',
                                    size=(.006*float(len(msg)), .02),
                                    font="Calco",
                                    justify='left')
                    box_actors[ids[i]] = vtk_box

                pts_3d_cam = get_box_points(box,
                                            point_num=8,
                                            show_box_heading=True)
                pts_3d_cam = box_op_velo_to_cam(pts_3d_cam[:,0:3],self.dataset.V2C)
                all_img_pts = np.matmul(pts_3d_cam, self.dataset.P2.T)  # (N, 3)
                #filter out targets with z less than 0
                show_index = np.where(all_img_pts[:, 2] > 0)[0]
                img_pts = all_img_pts[show_index]
                x, y = img_pts[:, 0] / img_pts[:, 2], img_pts[:, 1] / img_pts[:, 2]
                if len(x) <= 0:
                        continue
                x = np.clip(x, 2, W-2)
                y = np.clip(y, 2, H-2)
                x = x.astype(np.int32)
                y = y.astype(np.int32)
                xy = np.stack((x,y),1)
                hull = cv2.convexHull(np.array(xy))[:,0,:]

                cv2.drawContours(dst_color, [hull], 0, this_c[::-1], 3)
                cv2.drawContours(vis_instance, [hull], 0, this_c[::-1], -1)
                cv2.drawContours(dynamic_mask, [hull], 0, 1 if is_dynamic else 0, -1)

            #dst_color = cv2.addWeighted(dst_color,1., , 1.,0.)
            dst_color[dynamic_mask>0,2] = 255
            dst = cv2.vconcat( [dst_color,
                                vis_instance,
                                255*np.stack((dynamic_mask,dynamic_mask,dynamic_mask),axis=2)
                                ] )
            cv2.line(dst, (0,2*dst_color.shape[0]), (dst_color.shape[1],2*dst_color.shape[0]), (255,255,255), 2)
            cv2.imshow("dst", cv2.pyrDown(dst))
            #cv2.imshow("dynamic_mask", 255*dynamic_mask)


        if self.plt is not None:
            if hasattr(self, 'txt_curr'):
                self.plt.remove(self.txt_curr)
            self.txt_curr = vedo.Text2D("#%d/%d"%(self.i+1,len(self.dataset)), pos=(0.05, 0.95), s=1., c="black")

            #boxes_info, boxactors_for_id = self.get_3D_boxes(labels,ids=ids, show_ids=True,box_info=None)
            all_actors = [self.txt_curr, self.all_trj_actor]
            if curr_trj_actor is not None:
                all_actors.append(curr_trj_actor)
            all_actors += list(box_actors.values())

            x,y,z = self.trj[self.i,:]
            self.plt.show(all_actors,
                          interactive=False,
                          resetcam=False,
                          camera={'pos': (x, y-100, z-80), 'focalPoint': (x,y,z), 'viewup': (0, 0, 1)}
                          )
        self.i += 1
        return True

def Parse(dataset_path, dataset_type, seq):
    root = osp.join(dataset_path, dataset_type)
    label_path =osp.join(root, "label_02", seq+".txt")
    dataset = KittiTrackingDataset(root,seq,label_path)
    viewer = Parser(dataset)
    while viewer.do() :
        c = cv2.waitKey(1)
    return

if __name__ == '__main__':
    dataset_path ="./kitti_tracking_dataset"
    dataset_type = "training"
    seq = "0000"

    Parse(dataset_path, dataset_type, seq)
