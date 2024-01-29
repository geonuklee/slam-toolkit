import numpy as np
import vedo
from os import path as osp
import cv2
import vtk
from dataset.kitti_dataset import *
from viewer.color_map import generate_objects_color_map,generate_objects_colors,generate_scatter_colors
from viewer.box_op import convert_box_type,get_line_boxes,get_mesh_boxes,velo_to_cam,get_box_points
from scipy.spatial.transform import Rotation as rotation_util

class GlobalViewer:
    """
    Trajectory와 global 좌표상의 obj box를 표시하는 viewer
    default box type: "OpenPCDet", (x,y,z,l,w,h,yaw)
    """
    def __init__(self,dataset,box_type="Kitti",bg=(255, 255, 255)):
        self.objects_color_map = generate_objects_color_map('rainbow')
        self.box_type = box_type
        self.plt = vedo.Plotter(bg=bg)
        self.dataset = dataset
        poses = dataset.poses
        trj = []
        for pose in poses:
            trj.append( pose[:3] ) # T_w_c2's transloation
        self.trj = np.array(trj) # N by 3
        self.all_trj_actor = vedo.Line(self.trj, lw=1,c=(.5,.5,.5))

        self.T0_for_id = {}
        self.set_dynamic_obj = set()

        # Set the camera position to achieve the bird's-eye view
        self.reset_camera()
        self.plt.add_callback('KeyPress', self.keypress)
        dt_msec, self.animation_playing = 1, False
        self.button = self.plt.add_button(self.on_button_press, states=["Play", "Pause"], pos=(.1,.1))
        self.timerevt = self.plt.interactor.AddObserver('TimerEvent', self.on_timer)
        self.timer_id = self.plt.interactor.CreateRepeatingTimer(dt_msec)
        self.i = 0
        #self.on_button_press() # For auto start

    def reset_camera(self):
        self.plt.camera.SetPosition(0, -100, 0)
        self.plt.camera.SetFocalPoint(0, 0, 0)
        self.plt.camera.SetViewUp(0, 0, 1)
        self.plt.reset_camera()
        self.plt.render()
        return

    def on_button_press(self, widget=None, event=None):
        self.button.switch()
        self.animation_playing = not self.animation_playing
        return

    def exec(self):
        self.plt.show()
        return

    def keypress(self, evt):
        #print(evt.keypress)
        if evt.keypress == 's':
            self.on_button_press()
        if evt.keypress == 'r':
            self.reset_camera()
        if evt.keypress == 'q':
            self.plt.close()
        return

    def on_timer(self, iren, event):
        if not self.animation_playing:
            c = cv2.waitKey(1)
            if c == ord('q'):
                self.plt.close()
            return
        if self.i == self.trj.shape[0]-1:
            return
        self.plt.clear()

        # animate your stuff here
        print('TimerEvent', self.i)
        points, image, r_image, oxt, pose, labels, label_names = self.dataset[self.i]

        #cv2.imshow("image", cv2.pyrDown(image))
        dst_color = image.copy()
        vis_instance = np.zeros_like(dst_color)
        dynamic_mask = np.zeros(image.shape[:2],np.uint8)

        if self.i > 0:
            curr_trj_actor = vedo.Line(self.trj[:self.i,:], lw=2,c=(1.,0.,0.))
        else:
            curr_trj_actor = None

        # * [x] text_actor 를 박스위에 추가.
        # * [x] car 이외의 obj도 표시.
        # * [x] dynamic 판정된 instance를 cv image에 표시.
        # * [ ] evaluation
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
            T_w_c2[:3,:3] = rotation_util.from_quat( pose[3:] ).as_matrix()
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
                T_v_b[:3,:3] = rotation_util.from_euler('z',rad).as_matrix()
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

                #color = [rgb[-1],rgb[1],rgb[0]]
                pts_3d_cam = get_box_points(box,
                                            point_num=8,
                                            show_box_heading=True)
                pts_3d_cam = velo_to_cam(pts_3d_cam[:,0:3],self.dataset.V2C)
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
        #import pdb; pdb.set_trace()
        c = cv2.waitKey(1)
        if c == ord('q'):
            self.plt.close()
        elif c == ord('s'):
            self.on_button_press()
        self.i += 1
        return

def example():
    root="kitti_tracking_dataset/training"
    seq = "0000" # static
    #seq = "0001" # static
    #seq = "0003" # dynamic
    #seq = "0004"
    label_path =osp.join(root, "label_02", seq+".txt")
    dataset = KittiTrackingDataset(root,seq,label_path)
    viewer = GlobalViewer(dataset)
    viewer.exec()

if __name__ == '__main__':
    example()
