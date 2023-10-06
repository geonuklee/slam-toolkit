#include "seg_viewer.h"
#include <g2o/types/slam3d/se3quat.h>
#include <opencv2/highgui.hpp>
#include <pangolin/display/display.h>
#include <pangolin/display/view.h>
#include <pangolin/scene/axis.h>


SegViewer::SegViewer(const EigenMap<int, g2o::SE3Quat>& gt_Tcws, std::string config_fn, cv::Size dst_size)
  : name_ ("SegViewer"),
  gt_Tcws_(gt_Tcws),
  req_exit_(false)
{
  dst_size_ = dst_size;
  curr_k_ = 0;
  cv::FileStorage fsettings(config_fn, cv::FileStorage::READ);
  trj_size_.width   = fsettings["Viewer.width"];
  trj_size_.height  = fsettings["Viewer.height"];
  vp_f_         = fsettings["Viewer.vp_f"];
  z_near_       = fsettings["Viewer.z_near"];
  z_far_        = fsettings["Viewer.z_far"];
  ex_         = fsettings["Viewer.ex"];
  ey_         = fsettings["Viewer.ey"];
  ez_         = fsettings["Viewer.ez"];
  lx_         = fsettings["Viewer.lx"];
  ly_         = fsettings["Viewer.ly"];
  lz_         = fsettings["Viewer.lz"];
  ux_         = fsettings["Viewer.ux"];
  uy_         = fsettings["Viewer.uy"];
  uz_         = fsettings["Viewer.uz"];
  fps_          = 30.;
  thread_ = std::thread([&]() { Run(); });
}

void SegViewer::SetCurrCamera(int k, const g2o::SE3Quat& Tcw, const cv::Mat& dst) {
  std::unique_lock<std::mutex> lock(mutex_viewer_);
  curr_k_ = k;
  est_Tcws_[k] = Tcw;
  curr_dst_ = dst;
}

void SegViewer::SetMappoints(const EigenMap<int, Eigen::Vector3d>& mappoints) {
  for(auto it: curr_mappoints_)
    all_mappoints_[it.first] = it.second;
  curr_mappoints_ = mappoints;
  return;
}

pangolin::OpenGlMatrix Convert(const g2o::SE3Quat& _T);

void SegViewer::Join(bool req_exit) {
  if(req_exit){
      std::unique_lock<std::mutex> lock(mutex_viewer_);
      req_exit_ = req_exit;
  }
  if( thread_.joinable() )
    thread_.join();
  return;
}

void SegViewer::Run(){
  const int menu_width = 150;
  int height = std::max(trj_size_.height, dst_size_.height);
  pangolin::CreateWindowAndBind(name_, dst_size_.width+trj_size_.width+menu_width, height);

  glEnable(GL_DEPTH_TEST);
  glEnable (GL_BLEND);
  glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glPixelStorei(GL_UNPACK_ALIGNMENT,1); //  For GlTexture from cv::Mat

  pangolin::CreatePanel("menu").SetBounds(0.0,1.0,0.0,pangolin::Attach::Pix(menu_width));
  pangolin::Var<bool> menu_follow_camera("menu.Follow cam",true,true);
  pangolin::Var<bool> menu_show_points(  "menu.Show points",false,true);

  bool bFollow = true;
  bool bLocalizationMode = false;
  auto projection_matrix = pangolin::ProjectionMatrix(trj_size_.width, trj_size_.height,
                                                      vp_f_, vp_f_,
                                                      trj_size_.width/2.,trj_size_.height/2., z_near_,z_far_);
  auto lookat = pangolin::ModelViewLookAt(ex_, ey_, ez_, lx_, ly_, lz_, ux_, uy_, uz_);
  //auto lookat =  pangolin::ModelViewLookAt(1,0.5,-2,0,0,0, pangolin::AxisNegY);
  pangolin::OpenGlRenderState s_cam(projection_matrix, lookat);

  pangolin::View& d_cam = pangolin::Display("cam1")
    .SetBounds(0.0, 1.0, pangolin::Attach::Pix(menu_width), pangolin::Attach::Pix(menu_width+trj_size_.width) )
    .SetAspect(-float(trj_size_.width)/float(trj_size_.height))
    .SetHandler(new pangolin::Handler3D(s_cam));

  pangolin::View& d_img1 = pangolin::Display("img1")
    .SetBounds(0., 1., pangolin::Attach::Pix(menu_width+trj_size_.width), 1.)
    .SetAspect(-float(dst_size_.width)/float(height) );

  pangolin::Display("multi")
      .AddDisplay(d_cam)
      .AddDisplay(d_img1)
      ;

  pangolin::OpenGlMatrix gl_Twc;
  gl_Twc.SetIdentity();
  s_cam.Follow(gl_Twc);
  pangolin::GlTexture img_texture(dst_size_.width,dst_size_.height,GL_RGB,false,0,GL_RGB,GL_UNSIGNED_BYTE);

  //pangolin::RegisterKeyPressCallback(pangolin::PANGO_CTRL + 'b', [&](){
  pangolin::RegisterKeyPressCallback('q', [&](){
                                     std::cout << "Exit!" << std::endl;
                                     req_exit_ = true;
                                     });

  while( !pangolin::ShouldQuit() ) {
    // Sync member variables
    int curr_k;
    bool req_exit;
    cv::Mat curr_dst;
    EigenMap<int,g2o::SE3Quat> est_Tcws; {
      std::unique_lock<std::mutex> lock(mutex_viewer_);
      curr_k = curr_k_;
      req_exit = req_exit_;
      est_Tcws = est_Tcws_;
      cv::flip(curr_dst_, curr_dst,0);
    }
    if(req_exit)
      break;

    // After snyc 
    g2o::SE3Quat Twc;
    if(!est_Tcws.empty()){
      Twc = est_Tcws.at(curr_k).inverse();
    }
    auto t = Twc.translation();
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    gl_Twc = Convert(Twc);
    if(menu_follow_camera && bFollow) {
      s_cam.Follow(gl_Twc);
    }
    else if(menu_follow_camera && !bFollow) {
      s_cam.SetModelViewMatrix(lookat);
      s_cam.Follow(gl_Twc);
      bFollow = true;
    }
    else if(!menu_follow_camera && bFollow) {
      bFollow = false;
    }
    glClearColor(0.0f,0.0f,0.0f,1.0f);
    d_cam.Activate(s_cam);
    DrawTrajectories(est_Tcws);
    if(menu_show_points)
      DrawPoints();
    DrawPose(Twc);
    d_img1.Activate();
    if(!curr_dst.empty()){
      img_texture.Upload(curr_dst.data,GL_BGR,GL_UNSIGNED_BYTE);
      glColor4f(1.0f,1.0f,1.0f,1.0f);
      img_texture.RenderToViewport();
    }

    pangolin::FinishFrame();
    usleep(1e+6/fps_); // micro sec 
  }

  //auto view_mat = s_cam.GetModelViewMatrix();
  //view_mat.m; // TODO Save?
  pangolin::DestroyWindow(name_);
  return;
}

pangolin::OpenGlMatrix Convert(const g2o::SE3Quat& _T){
  pangolin::OpenGlMatrix T;
  Eigen::Matrix4f mat = _T.to_homogeneous_matrix().cast<float>();
  T.m[0] = mat(0,0);
  T.m[1] = mat(1,0);
  T.m[2] = mat(2,0);
  T.m[3]  = 0.0;
  T.m[4] = mat(0,1);
  T.m[5] = mat(1,1);
  T.m[6] = mat(2,1);
  T.m[7]  = 0.0;
  T.m[8]  = mat(0,2);
  T.m[9]  = mat(1,2);
  T.m[10] = mat(2,2);
  T.m[11]  = 0.0;
  T.m[12] = mat(0,3);
  T.m[13] = mat(1,3);
  T.m[14] = mat(2,3);
  T.m[15]  = 1.0;
  return T;
}
void SegViewer::DrawPose(const g2o::SE3Quat& _Twc) {
  auto Twc = Convert(_Twc);
  const float w = 1.; // configureation 불러오기 추가필요..
  const float h = w*0.75;
  const float z = w*0.6;

  glPushMatrix();
#ifdef HAVE_GLES
  glMultMatrixf(Twc.m);
#else
  glMultMatrixd(Twc.m);
#endif
  glLineWidth(5);

#if 0
  glColor4f(1., 0., 0.,1.);
  glBegin(GL_LINES);
  glVertex3f(0,0,0);
  glVertex3f(w,0,0);

  glColor4f(0, 1., 0.,1.);
  glVertex3f(0,0,0);
  glVertex3f(0,w,0);

  glColor4f(0, 0., 1.,1.);
  glVertex3f(0,0,0);
  glVertex3f(0,0,w);
  glEnd();

#else
  glLineWidth(2);
  glColor4f(0., 0., 1.,1.);

  glBegin(GL_LINES);
  glVertex3f(0,0,0);
  glVertex3f(w,h,z);
  glVertex3f(0,0,0);
  glVertex3f(w,-h,z);
  glVertex3f(0,0,0);
  glVertex3f(-w,-h,z);
  glVertex3f(0,0,0);
  glVertex3f(-w,h,z);

  glVertex3f(w,h,z);
  glVertex3f(w,-h,z);

  glVertex3f(-w,h,z);
  glVertex3f(-w,-h,z);

  glVertex3f(-w,h,z);
  glVertex3f(w,h,z);

  glVertex3f(-w,-h,z);
  glVertex3f(w,-h,z);
  glEnd();
#endif

  glPopMatrix();
}

void SegViewer::DrawTrajectories(const EigenMap<int,g2o::SE3Quat>& est_Tcws) {
  int curr_k = est_Tcws.empty()? 0 : est_Tcws.rbegin()->first;
  if(!gt_Tcws_.empty()){ // Draw ground truth trajectories
    for(size_t i=0; i+1 < gt_Tcws_.size(); i++){
      glLineWidth(1);
      if(i >= curr_k)
        glColor4f(.3,.3,.3,1.);
      else
        glColor4f(.6,.6,.6,1.);
      glBegin(GL_LINES);
      const g2o::SE3Quat Twc0 = gt_Tcws_.at(i).inverse();
      const g2o::SE3Quat Twc1 = gt_Tcws_.at(i+1).inverse();
      const auto& t0 = Twc0.translation();
      const auto& t1 = Twc1.translation();
      glVertex3f(t0.x(), t0.y(), t0.z());
      glVertex3f(t1.x(), t1.y(), t1.z());
      glEnd();
    }
  }

  if(!est_Tcws.empty()){ // Draw ground truth trajectories
    glColor4f(0.,1.,0.,1.);
    glLineWidth(1);
    glPointSize(3.);
    glBegin(GL_LINES);
    for(size_t i=0; i+1 < est_Tcws.size(); i++){
      const g2o::SE3Quat Twc0 = est_Tcws.at(i).inverse();
      const g2o::SE3Quat Twc1 = est_Tcws.at(i+1).inverse();
      const auto& t0 = Twc0.translation();
      const auto& t1 = Twc1.translation();
      glVertex3f(t0.x(), t0.y(), t0.z());
      glVertex3f(t1.x(), t1.y(), t1.z());
    }
    glEnd();
  }
  return;
}

void SegViewer::DrawPoints() {
  glBegin(GL_POINTS);
  glPointSize(.3);
  glColor4f(.5,.5,.5,1.);
  for(auto it : all_mappoints_){
    if(curr_mappoints_.count(it.first) )
      continue;
    glVertex3f(it.second.x(), it.second.y(), it.second.z() );
  }
  glPointSize(1.);
  glColor4f(1.,1.,0.,1.);
  for(auto it : curr_mappoints_)
    glVertex3f(it.second.x(), it.second.y(), it.second.z() );
  glEnd();
}

void setImageData(unsigned char * imageArray, int size){
  for(int i = 0 ; i < size;i++) {
    imageArray[i] = (unsigned char)(rand()/(RAND_MAX/255.0));
  }
}

void TestPangolin(int argc, char** argv) {
  std::string rgb_fn = std::string(PACKAGE_DIR)+"/kitti_tracking_dataset/training/image_02/0000/000000.png";
  cv::Mat rgb = cv::imread(rgb_fn);
  int im_width = rgb.cols;
  int trj_width = 300.;
  int height = rgb.rows;
  float vp_f = 400;
  float near = 0.01;
  float far = 1000.;
  float fps = 20.;
  int menu_width = 140;

  pangolin::CreateWindowAndBind("pangolin", im_width+trj_width+menu_width, height);
  glEnable(GL_DEPTH_TEST);
  glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  pangolin::CreatePanel("menu").SetBounds(0.0,1.0,0.0,pangolin::Attach::Pix(menu_width));
  pangolin::Var<bool> menu_follow_camera("menu.Follow cam",true,true);
  pangolin::Var<bool> menu_show_points("menu.Show pt",false,true);

  pangolin::OpenGlMatrix proj = pangolin::ProjectionMatrix(trj_width,height,420,420,trj_width/2.,height/2.,0.1,1000);
  pangolin::OpenGlRenderState s_cam(proj, pangolin::ModelViewLookAt(1,0.5,-2,0,0,0, pangolin::AxisY) );

  pangolin::Handler3D handler(s_cam);
  pangolin::View& d_cam = pangolin::Display("cam1")
    .SetBounds(0., 1., pangolin::Attach::Pix(menu_width), pangolin::Attach::Pix(menu_width+trj_width))
    .SetAspect(-float(trj_width)/float(height) )
    .SetHandler(new pangolin::Handler3D(s_cam));

  pangolin::View& d_img1 = pangolin::Display("img1")
    .SetBounds(0., 1., pangolin::Attach::Pix(menu_width+trj_width), 1.)
    .SetAspect( float(im_width)/float(height) );

  pangolin::Display("multi")
      .AddDisplay(d_img1)
      .AddDisplay(d_cam)
      ;

  // Issue specific OpenGl we might need
  glEnable (GL_BLEND);
  glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  glPixelStorei(GL_UNPACK_ALIGNMENT,1);
  pangolin::GlTexture imageTexture(im_width,height,GL_RGB,false,0,GL_RGB,GL_UNSIGNED_BYTE);

  while( !pangolin::ShouldQuit() ) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glClearColor(0.0f,0.0f,0.0f,1.0f);
    d_cam.Activate(s_cam);
    pangolin::glDrawColouredCube();

    //setImageData(imageArray,3*im_width*height);
    cv::Mat fliped_rgb;
    cv::flip(rgb, fliped_rgb,0);
    imageTexture.Upload(fliped_rgb.data,GL_BGR,GL_UNSIGNED_BYTE);
    d_img1.Activate();
    glColor4f(1.0f,1.0f,1.0f,1.0f);
    imageTexture.RenderToViewport();

    pangolin::FinishFrame();
    usleep(1e+4);
  }

  return;
}


