#include "seg_dataset.h"
#include "camera.h"
#include "hitnet.h"
#include <filesystem>

void WriteKittiTrajectory(const g2o::SE3Quat& Tcw,
                          std::ofstream& output_file) {
  output_file << std::scientific;
  g2o::SE3Quat Twc = Tcw.inverse();
  Eigen::Matrix<double,3,4> Rt = Twc.to_homogeneous_matrix().block<3,4>(0,0).cast<double>();
  for(size_t i = 0; i < 3; i++){
    for(size_t j = 0; j < 4; j++){
      output_file << Rt(i,j);
      if(i==2 && j==3)
        continue;
      output_file << " ";
    }
  }
  output_file << std::endl;
  output_file.flush();
  return;
}



// https://stackoverflow.com/questions/17735863/opencv-save-cv-32fc1-images
bool writeRawImage(const cv::Mat& image, const std::string& filename) {
    std::ofstream file(filename, std::ios::out | std::ios::binary);
    if (!file.is_open())
        return false;

    int rows = image.rows;
    int cols = image.cols;
    int channels = image.channels();
    int sizeInBytes = image.total() * image.elemSize(); // Calculate size based on total elements

    file.write(reinterpret_cast<const char*>(&rows), sizeof(int));
    file.write(reinterpret_cast<const char*>(&cols), sizeof(int));
    file.write(reinterpret_cast<const char*>(&channels), sizeof(int));
    file.write(reinterpret_cast<const char*>(&sizeInBytes), sizeof(int));
    file.write(reinterpret_cast<const char*>(image.data), sizeInBytes);

    file.close();
    return true;
}

cv::Mat readRawImage(const std::string& filename, int type) {
    cv::Mat image;

    std::ifstream file(filename, std::ios::in | std::ios::binary);
    if (!file.is_open())
        return image;

    int rows, cols, channels, sizeInBytes;

    file.read(reinterpret_cast<char*>(&rows), sizeof(int));
    file.read(reinterpret_cast<char*>(&cols), sizeof(int));
    file.read(reinterpret_cast<char*>(&channels), sizeof(int));
    file.read(reinterpret_cast<char*>(&sizeInBytes), sizeof(int));

    if (rows < 1 || cols < 1 || channels < 1 || sizeInBytes < 1) {
        file.close();
        return image;
    }

    image = cv::Mat(rows, cols, type);
    file.read(reinterpret_cast<char*>(image.data), sizeInBytes);

    file.close();
    return image;
}

EigenVector<g2o::SE3Quat> ParseOxts(std::string fn_oxts) {
/*
Generator to read OXTS ground truth data.

Poses are given in an East-North-Up coordinate system 
whose origin is the first GPS position.

Original codes : 
https://github.com/utiasSTARS/pykitti/blob/0.3.1/pykitti/utils.py#L12
https://github.com/utiasSTARS/pykitti/blob/0.3.1/pykitti/utils.py#L85

input: oxts txt path
*/
  double er = 6378137.;  // earth radius (approx.) in meters
  std::ifstream fs;
  fs.open(fn_oxts);
  std::string str;
  int n = 0;
  EigenVector<g2o::SE3Quat> vec_TWi;
  double scale = -1.;
  while( std::getline(fs, str) ){
   double lat, lon, alt, \
           roll, pitch, yaw, \
           vn, ve, vf, vl, vu, \
           ax, ay, az, af, al, au, \
           wx, wy, wz, wf, wl, wu, \
           pos_accuracy, vel_accuracy;
    int navstat, numsats, posmode, velmode, orimode;
    sscanf(str.c_str(),"%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %d %d %d %d %d", 
           &lat, &lon, &alt,
           &roll, &pitch, &yaw,
           &vn, &ve, &vf, &vl, &vu, 
           &ax, &ay, &az, &af, &al, &au,
           &wx, &wy, &wz, &wf, &wl, &wu,
           &pos_accuracy, &vel_accuracy,
           &navstat, &numsats, &posmode, &velmode, &orimode
           );
    if(scale < 0.)
      scale = std::cos(lat * M_PI / 180.);
    double tx = scale * lon * M_PI * er / 180.;
    double ty = scale * er * std::log( std::tan( (90.+lat)*M_PI/ 360. ) );
    double tz = alt;
    Eigen::Vector3d t(tx,ty,tz);
    Eigen::Matrix3d R_roll, R_pitch, R_yaw;
    R_roll << 1, 0, 0,
           0, cos(roll), -sin(roll),
           0, sin(roll), cos(roll);
    R_pitch << cos(pitch), 0, sin(pitch),
            0, 1, 0,
            -sin(pitch), 0, cos(pitch);
    R_yaw << cos(yaw), -sin(yaw), 0,
          sin(yaw), cos(yaw), 0,
          0, 0, 1;
    Eigen::Matrix3d R = R_roll * R_yaw * R_pitch;
    g2o::SE3Quat TWi(R,t);
    vec_TWi.push_back(TWi);
  }
  return vec_TWi;
}

KittiTrackingDataset::KittiTrackingDataset(std::string type, //  "training"|"testing"
                                           std::string seq,
                                           std::string dataset_path)
  : camera_(nullptr)
{
  if(dataset_path.empty())
    dataset_path = GetPackageDir() + "/kitti_tracking_dataset";
  if(dataset_path.back() == '/')
    dataset_path.pop_back();
  dataset_path_ = dataset_path;
  type_ = type;
  seq_ = seq;

  std::string im0_path   = dataset_path+"/" + type + "/image_02/" + seq;
  std::string im1_path   = dataset_path+"/" + type + "/image_03/" + seq;
  depth_path_ = dataset_path+"/" + type + "/depth_02/";
  mask_path_ = dataset_path+"/" + type + "/mask_02/";

  std::filesystem::directory_iterator it{im0_path}, end;
  for(; it != end; it++){
    std::string str = it->path().filename().stem().string();
    if(it->path().filename().extension() != ".png"){
      std::cout << it->path().filename() << std::endl;
      std::cout << it->path().filename().extension() << std::endl;
      std::cout << "not png" << std::endl;
      exit(1);
    }
    size_t i = std::stoi(str);
    im0_filenames_[i] = it->path().string();
    if( std::filesystem::exists(im1_path) )
      im1_filenames_[i] = im1_path+"/" + it->path().filename().string();
  }

  std::string fn_calib = dataset_path+"/" + type + "/calib/" + seq + ".txt";
  g2o::SE3Quat T_i_c2;
  {
    std::ifstream fs;
    fs.open(fn_calib);
    std::string str;
    Eigen::Matrix<double,3,4> P0, P1, P2, P3;
    Eigen::Matrix<double,3,3> R_rect;
    g2o::SE3Quat T_r_v, T_v_i;
    {
      std::getline(fs, str);
      float elem[12];
      if(!sscanf(str.c_str(),"P0: %f %f %f %f %f %f %f %f %f %f %f %f  \n", elem, elem+1, elem+2, elem+3, elem+4, elem+5, elem+6, elem+7, elem+8, elem+9, elem+10, elem+11) )
        throw -1;
      P0 = Eigen::Matrix<float,3,4,Eigen::RowMajor>(elem).cast<double>();
    }
    {
      std::getline(fs, str);
      float elem[12];
      if(!sscanf(str.c_str(),"P1: %f %f %f %f %f %f %f %f %f %f %f %f  \n", elem, elem+1, elem+2, elem+3, elem+4, elem+5, elem+6, elem+7, elem+8, elem+9, elem+10, elem+11) )
        throw -1;
      P1 = Eigen::Matrix<float,3,4,Eigen::RowMajor>(elem).cast<double>();
    }
    {
      std::getline(fs, str);
      float elem[12];
      if(!sscanf(str.c_str(),"P2: %f %f %f %f %f %f %f %f %f %f %f %f  \n", elem, elem+1, elem+2, elem+3, elem+4, elem+5, elem+6, elem+7, elem+8, elem+9, elem+10, elem+11) )
        throw -1;
      P2 = Eigen::Matrix<float,3,4,Eigen::RowMajor>(elem).cast<double>();
    }
    {
      std::getline(fs, str);
      float elem[12];
      if(!sscanf(str.c_str(),"P3: %f %f %f %f %f %f %f %f %f %f %f %f  \n", elem, elem+1, elem+2, elem+3, elem+4, elem+5, elem+6, elem+7, elem+8, elem+9, elem+10, elem+11) )
        throw -1;
      P3 = Eigen::Matrix<float,3,4,Eigen::RowMajor>(elem).cast<double>();
    }
    {
      std::getline(fs, str);
      float elem[9];
      if(!sscanf(str.c_str(),"R_rect %f %f %f %f %f %f %f %f %f %f %f %f  \n", elem, elem+1, elem+2, elem+3, elem+4, elem+5, elem+6, elem+7, elem+8, elem+9, elem+10, elem+11) )
        throw -1;
      R_rect = Eigen::Matrix<float,3,3,Eigen::RowMajor>(elem).cast<double>();
    }
    {
      std::getline(fs, str);
      float elem[12];
      if(!sscanf(str.c_str(),"Tr_velo_cam %f %f %f %f %f %f %f %f %f %f %f %f  \n", elem, elem+1, elem+2, elem+3, elem+4, elem+5, elem+6, elem+7, elem+8, elem+9, elem+10, elem+11) )
        throw -1;
      Eigen::Matrix<double,3,4> Tr_velo_cam = Eigen::Matrix<float,3,4,Eigen::RowMajor>(elem).cast<double>();
      T_r_v = g2o::SE3Quat(Tr_velo_cam.block<3,3>(0,0), Tr_velo_cam.block<3,1>(0,3) );
    }
    {
      std::getline(fs, str);
      float elem[12];
      if(!sscanf(str.c_str(),"Tr_imu_velo %f %f %f %f %f %f %f %f %f %f %f %f  \n", elem, elem+1, elem+2, elem+3, elem+4, elem+5, elem+6, elem+7, elem+8, elem+9, elem+10, elem+11) )
        throw -1;
      Eigen::Matrix<double,3,4> Tr_imu_velo = Eigen::Matrix<float,3,4,Eigen::RowMajor>(elem).cast<double>();
      T_v_i = g2o::SE3Quat(Tr_imu_velo.block<3,3>(0,0), Tr_imu_velo.block<3,1>(0,3) );
    }
    /*
    std::cout << "P0 = \n" << P0 << std::endl;
    std::cout << "P1 = \n" << P1 << std::endl;
    std::cout << "P2 = \n" << P2 << std::endl;
    std::cout << "P3 = \n" << P3 << std::endl;
    */
    g2o::SE3Quat T_c0_v = g2o::SE3Quat(R_rect, Eigen::Vector3d(0.,0.,0.)) * T_r_v;
    g2o::SE3Quat T_v_c0 = T_c0_v.inverse();
    g2o::SE3Quat T_i_v = T_v_i.inverse();
    Eigen::Vector3d t_c2_c0 = P2.block<3,3>(0,0).inverse() * P2.block<3,1>(0,3);
    Eigen::Vector3d t_c3_c0 = P3.block<3,3>(0,0).inverse() * P3.block<3,1>(0,3);
    double base_line = (t_c2_c0 - t_c3_c0)[0]; // or norm
    g2o::SE3Quat T_c0_c2;
    T_c0_c2.setTranslation(-t_c2_c0);
    T_i_c2 = T_i_v * T_c0_v.inverse() * T_c0_c2 ;

    const Eigen::Matrix3d K = P2.block<3,3>(0,0);
    const Eigen::Vector4d D(0.,0.,0.,0.); // Rectified
    g2o::SE3Quat Trl; // r=={c3}, l=={c2}
    Trl.setTranslation(t_c3_c0-t_c2_c0);
    cv::Mat im0 = cv::imread(im0_filenames_[0]);
    camera_ = new StereoCamera(K, D, K, D, Trl, im0.cols, im0.rows);
  }
  std::string fn_oxts = dataset_path+"/" + type + "/oxts/" + seq + ".txt";
  EigenVector<g2o::SE3Quat> vec_TWi = ParseOxts(fn_oxts);
  g2o::SE3Quat T_w_W;
  for(size_t i=0; i < vec_TWi.size(); i++){
    const auto& T_W_i = vec_TWi.at(i);
    g2o::SE3Quat T_W_c2 =  T_W_i * T_i_c2;
    if(i==0)
      T_w_W = T_W_c2.inverse(); // 최초의 c2를 원점으로.
    Tcws_[i] = (T_w_W * T_W_c2).inverse();
  }
} // KittiTrackingDataset::KittiTrackingDataset()

KittiTrackingDataset::~KittiTrackingDataset() {
  if(camera_)
    delete camera_;
}

cv::Mat KittiTrackingDataset::GetImage(int i, int flags) const{
  std::string fn = im0_filenames_.at(i);
  cv::Mat src = cv::imread(fn, flags);
  return src;
}

cv::Mat KittiTrackingDataset::GetRightImage(int i, int flags) const{
  std::string fn = im1_filenames_.at(i);
  cv::Mat src = cv::imread(fn, flags);
  return src;
}

bool KittiTrackingDataset::EixstCachedDepthImages() const {
  return !GetDepthImage(this->Size()-1).empty();
}

cv::Mat KittiTrackingDataset::GetDepthImage(int i) const{
#if 1
  char buffer[24];
  std::sprintf(buffer, "/%06d.raw", i);
  std::string fn = depth_path_+seq_+std::string(buffer);
  cv::Mat depth = readRawImage(fn, CV_32FC1);
#else
  std::string fn = depth_filenames_.at(i);
  std::ifstream file(fn, std::ios::binary | std::ios::ate);
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);
  char* buffer = new char[size];
  if (!file.read(buffer, size)) {
    std::cerr << "Failed to read file.";
    return cv::Mat();
  }
  file.close();
  int32_t rows = *(int32_t*)buffer;
  int32_t cols = *(int32_t*)(buffer + sizeof(int32_t));
  float* data = (float*)(buffer + 2 * sizeof(int));
  cv::Mat depth(rows,cols,CV_32FC1,data); // delete data는 필요없음.
#endif
  return depth;
}

cv::Mat KittiTrackingDataset::GetDynamicMask(int i) const {
  char buffer[24];
  std::sprintf(buffer, "/dynamic%06d.png", i);
  std::string fn = mask_path_+seq_+std::string(buffer);
  return cv::imread(fn, cv::IMREAD_GRAYSCALE);
}

cv::Mat KittiTrackingDataset::GetInstanceMask(int i) const {
  char buffer[24];
  std::sprintf(buffer, "/ins%06d.raw", i);
  std::string fn = mask_path_+seq_+std::string(buffer);
  return readRawImage(fn, CV_32SC1);
}


int KittiTrackingDataset::Size() const{
  return im0_filenames_.size();
}

const EigenMap<int, g2o::SE3Quat>& KittiTrackingDataset::GetTcws() const {
  return Tcws_;
}

const Camera* KittiTrackingDataset::GetCamera() const {
  return camera_;
}

#if 1
#include <pybind11/embed.h>
#include "hitnet.h"
namespace py = pybind11;
namespace fs = std::filesystem;

void KittiTrackingDataset::ComputeCacheImages() { // depth image, dynamic instance mask.
  std::string pkg_dir = PACKAGE_DIR;
  std::string pycode;
  pycode += "import sys; sys.path.append(\"" + pkg_dir + "\")\n";
  pycode += "import kitti_tracking_tools as kt\n";
  pycode += "kt.Parse(\"" + pkg_dir + "/kitti_tracking_dataset\", \""+type_+"\", \""+seq_+"\")\n";
  //std::cout << "----------------------" << std::endl;
  //std::cout << pycode << std::endl;
  //std::cout << "----------------------" << std::endl;
  py::exec(pycode);

  if( !fs::exists(depth_path_) ) 
    fs::create_directories(depth_path_);
  if( fs::exists(depth_path_+seq_) ) 
    fs::remove_all(depth_path_+seq_);
  fs::create_directories(depth_path_+seq_);

  StereoCamera* scam = static_cast<StereoCamera*>(camera_);
  float base_line = -scam->GetTrl().translation()[0];
  float min_disp = 1.;
  float fx = camera_->GetK()(0,0);
  HITNetStereoMatching hitnet;
  char buffer[24];
  for(size_t i =0; i < this->Size(); i++){
    cv::Mat gray = GetImage(i, cv::IMREAD_GRAYSCALE);
    cv::Mat gray_r = GetRightImage(i, cv::IMREAD_GRAYSCALE);
    cv::Mat disp  = hitnet.GetDisparity(gray, gray_r);
    cv::Mat depth = Disparit2Depth(disp,base_line,fx, min_disp);
    std::sprintf(buffer, "/%06ld.raw", i);
    std::string fn = depth_path_+seq_+ std::string(buffer);
    writeRawImage(depth, fn);
    cv::Mat load = readRawImage(fn, CV_32FC1);

    g2o::SE3Quat Twc = Tcws_.at(i).inverse();
    cv::imshow("gray", gray);
    cv::imshow("gray_r", gray_r);
    cv::imshow("disp", 0.01*disp);
    cv::imshow("depth", 0.01*load);
    char c = cv::waitKey(1);
    if(c == 'q')
      exit(1);
  }
  return;
}
#endif


#include "segslam.h"
namespace NEW_SEG {

EvalWriter::EvalWriter(std::string output_seq_dir) 
: output_seq_dir_(output_seq_dir),
  output_mask_dir_( output_seq_dir+"/"+"mask"),
  trj_output_(output_seq_dir+"/"+"trj.txt"),
  keypoints_output_(output_seq_dir+"/"+"keypoints.txt")
{
  if(std::filesystem::exists(output_mask_dir_) )
    std::filesystem::remove_all(output_mask_dir_);
  std::filesystem::create_directories(output_mask_dir_);
}

void EvalWriter::Write(Frame* frame,
                       RigidGroup* static_rig,
                       const cv::Mat synced_marker,
                       const std::set<int>& uniq_labels,
                       const cv::Mat gt_insmask,
                       const cv::Mat gt_dmask) {
  const Qth qth = static_rig->GetId();
  assert(qth==0);

  {
    // 1. Trajectory 
    g2o::SE3Quat Tcw = frame->GetTcq(qth);
    WriteKittiTrajectory(Tcw, trj_output_); 
  }

  int frame_id = frame->GetId();
  {
    /* 2. keypoints 저장.
    */
    const auto& keypoints = frame->GetKeypoints();
    const auto& mappoints = frame->GetMappoints();
    for(size_t n=0; n < keypoints.size(); n++){
      const cv::Point2f& pt = keypoints[n].pt;
      Mappoint* mp = mappoints[n];
      Instance* ins = mp ? mp->GetInstance() : nullptr;
      int has_mp = mp ? 1 : 0;
      int ins_id = ins ? ins->GetId() : 0;
      int est_on_dynamic = 0;
      if(ins && ins->GetId() != 0)
        est_on_dynamic = 1;
      int gt_on_dynamic =  gt_dmask.at<uchar>(pt) > 0 ? 1 : 0;
      keypoints_output_ << frame_id << " "
                        << n << " "
                        << ins_id << " "
                        << has_mp << " "
                        << pt.x << " "
                        << pt.y  << std::endl;
    }
    keypoints_output_.flush();
  }

  {
    // 3. output_mask_dir_ 에 dynamic%06d.png, ins%06d.raw 저장.
    cv::Mat est_dmask = cv::Mat::zeros(synced_marker.rows, synced_marker.cols, CV_8UC1);
    for(int pth : uniq_labels){
      bool est_on_dynamic = static_rig->GetExcludedInstances().count(pth);
      if(est_on_dynamic)
        est_dmask.setTo(255, synced_marker==pth);
    }

    char filename[50]; // A buffer to hold the formatted string
    // Format the string "mask%06d.png"
    std::sprintf(filename, "/dynamic%06d.png", frame_id);
    cv::imwrite(output_mask_dir_+std::string(filename), est_dmask);

    std::sprintf(filename, "/ins%06d.raw", frame_id);
    writeRawImage(synced_marker, output_mask_dir_+std::string(filename) );
  }
 
  return;
}

} // namespace NEW_SEG 
