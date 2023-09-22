#include "seg_viewer.h"
#include <g2o/types/slam3d/se3quat.h>
#include <pangolin/display/display.h>
#include <pangolin/scene/axis.h>


SegViewer::SegViewer(const EigenMap<int, g2o::SE3Quat>& gt_Tcws, std::string config_fn)
  : name_ ("SegViewer"),
  gt_Tcws_(gt_Tcws)
{
  curr_k_ = 0;
  cv::FileStorage fsettings(config_fn, cv::FileStorage::READ);
  size_.width   = fsettings["Viewer.width"];
  size_.height  = fsettings["Viewer.height"];
  vp_f_         = fsettings["Viewer.vp_f"];
  z_near_       = fsettings["Viewer.z_near"];
  z_far_        = fsettings["Viewer.z_far"];
  ex_         = fsettings["Viewer.ex"];
  ey_         = fsettings["Viewer.ey"];
  ez_         = fsettings["Viewer.ez"];
  lx_         = fsettings["Viewer.lx"];
  ly_         = fsettings["Viewer.ly"];
  lz_         = fsettings["Viewer.lz"];
  fps_          = 30.;
  thread_ = std::thread([&]() { Run(); });
}

void SegViewer::SetFrame(int k) {
  std::unique_lock<std::mutex> lock(mutex_viewer_);
  curr_k_ = k;
}

pangolin::OpenGlMatrix Convert(const g2o::SE3Quat& _T);

void SegViewer::Join() {
  thread_.join();
  return;
}

void SegViewer::Run(){
  pangolin::CreateWindowAndBind(name_, size_.width, size_.height);
  glEnable(GL_DEPTH_TEST);
  // Issue specific OpenGl we might need
  glEnable (GL_BLEND);
  glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  pangolin::CreatePanel("menu").SetBounds(0.0,1.0,0.0,pangolin::Attach::Pix(175));
  pangolin::Var<bool> menuFollowCamera("menu.Follow Camera",true,true);
  pangolin::Var<bool> menuShowPoints("menu.Show Points",true,true);
  pangolin::Var<bool> menuShowKeyFrames("menu.Show KeyFrames",true,true);
  pangolin::Var<bool> menuShowGraph("menu.Show Graph",true,true);
  pangolin::Var<bool> menuLocalizationMode("menu.Localization Mode",false,true);
  pangolin::Var<bool> menuReset("menu.Reset",false,false);
  bool bFollow = true;
  bool bLocalizationMode = false;
  auto projection_matrix = pangolin::ProjectionMatrix(size_.width, size_.height,
                                                      vp_f_, vp_f_,
                                                      size_.width/2.,size_.height/2., z_near_,z_far_);
  auto lookat = pangolin::ModelViewLookAt(ex_, ey_, ez_, lx_, ly_, lz_, 0.,-1.,0.);

  pangolin::OpenGlRenderState s_cam(projection_matrix,
                                    lookat);
  pangolin::Handler3D handler(s_cam);
  pangolin::View& d_cam = pangolin::CreateDisplay()
    .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -float(size_.width)/float(size_.height))
    .SetHandler(new pangolin::Handler3D(s_cam));

  // Issue specific OpenGl we might need
  glEnable (GL_BLEND);
  glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  pangolin::OpenGlMatrix gl_Twc;
  gl_Twc.SetIdentity();
  s_cam.Follow(gl_Twc);

  while( !pangolin::ShouldQuit() ) {
    // Sync member variables
    int curr_k; {
      std::unique_lock<std::mutex> lock(mutex_viewer_);
      curr_k = curr_k_;
    }

    // After snyc 
    g2o::SE3Quat Twc;
    // TODO gt 대신, given camera pose를 따라가도록 변경.
    if(!gt_Tcws_.empty())
      Twc = gt_Tcws_.at(curr_k).inverse();
    auto t = Twc.translation();
    gl_Twc = Convert(Twc);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    if(menuFollowCamera && bFollow) {
      s_cam.Follow(gl_Twc);
    }
    else if(menuFollowCamera && !bFollow) {
      s_cam.SetModelViewMatrix(lookat);
      s_cam.Follow(gl_Twc);
      bFollow = true;
    }
    else if(!menuFollowCamera && bFollow) {
      bFollow = false;
    }
    d_cam.Activate(s_cam);
    glClearColor(0.0f,0.0f,0.0f,1.0f);
    DrawPose(Twc);

    if(!gt_Tcws_.empty()){ // Draw ground truth trajectories
      glLineWidth(4);
      glColor3f(.6,.6,.6);
      glBegin(GL_LINES);
      for(size_t i=0; i+1 < gt_Tcws_.size(); i++){
        if(i >= curr_k){
          glLineWidth(2);
          glColor3f(.3,.3,.3);
        }
        const g2o::SE3Quat Twc0 = gt_Tcws_.at(i).inverse();
        const g2o::SE3Quat Twc1 = gt_Tcws_.at(i+1).inverse();
        const auto& t0 = Twc0.translation();
        const auto& t1 = Twc1.translation();
        glVertex3f(t0.x(), t0.y(), t0.z());
        glVertex3f(t1.x(), t1.y(), t1.z());
      }
      glEnd();
    }
    pangolin::FinishFrame();
  }
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

#if 1
  glColor3f(1., 0., 0.);
  glBegin(GL_LINES);
  glVertex3f(0,0,0);
  glVertex3f(w,0,0);

  glColor3f(0, 1., 0.);
  glVertex3f(0,0,0);
  glVertex3f(0,w,0);

  glColor3f(0, 0., 1.);
  glVertex3f(0,0,0);
  glVertex3f(0,0,w);
  glEnd();

#else
  glColor3f(0.0f,1.0f,0.0f);
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

