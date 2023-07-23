/*
Copyright (c) 2020 Geonuk Lee

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following
conditions:
The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.
*/

#include "dataset.h"
#include "orb_extractor.h"
#include <filesystem>

/*
ORB_SLAM2::ORBextractor* KittiDataset::GetExtractor() const {
  {
    int nfeatures = 2000;
    float scale_factor = 1.2;
    int nlevels = 8;
    int initial_fast_th = 20;
    int min_fast_th = 7;

    extractor_
      = new ORB_SLAM2::ORBextractor(nfeatures,
                                    scale_factor,
                                    nlevels,
                                    initial_fast_th,
                                    min_fast_th);
  }

  return extractor_;
}
*/

KittiDataset::KittiDataset(std::string seq, std::string dataset_path)
  : camera_(nullptr)
{
  if(dataset_path.empty())
    dataset_path = GetPackageDir() + "/kitti_odometry_dataset";
  std::string im0_path   = dataset_path+"/sequences/"+seq+"/image_0";
  std::string im1_path   = dataset_path+"/sequences/"+seq+"/image_1";
  std::string depth_path = dataset_path+"/sequences/"+seq+"/depth_0";
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
    if( std::filesystem::exists(depth_path) )
      depth_filenames_[i] = depth_path+"/" + it->path().filename().string();
  }

  const std::string fn_poses  = dataset_path + "/poses/"+seq+".txt";
  {
    std::ifstream fs;
    fs.open(fn_poses);
    std::string str;
    int n = 0;
    while( std::getline(fs, str) ){
      float elem[12];
      sscanf(str.c_str(),"%f %f %f %f %f %f %f %f %f %f %f %f \n", elem, elem+1, elem+2, elem+3, elem+4, elem+5, elem+6, elem+7, elem+8, elem+9, elem+10, elem+11);
      Eigen::Matrix<double,3,4> Rt; // Twc
      for(size_t i = 0; i < 3; i++)
        for(size_t j = 0; j < 4; j++)
          Rt(i,j) = elem[4*i+j];
      g2o::SE3Quat Twc;
      Eigen::AngleAxisd aa;
      aa.fromRotationMatrix(Rt.block<3,3>(0,0));
      Twc.setRotation(Eigen::Quaterniond(aa));
      Twc.setTranslation(Rt.block<3,1>(0,3));
      Tcws_[n++] = Twc.inverse();
    }
  }

  EigenMap<std::string, Eigen::Matrix<double,3,4> > Projections; {
    std::string calib_path = dataset_path+"/sequences/"+seq+"/calib.txt";
    std::ifstream file(calib_path);
    std::string line;

    while (std::getline(file, line)) {
      std::istringstream iss(line);
      std::string name;
      iss >> name;
      name = name.substr(0, name.size() - 1);
      Eigen::Matrix<double,3,4> P;
      for(int r=0; r<P.rows(); r++){
        for(int c=0; c<P.cols(); c++){
          double value;
          iss >> value;
          P(r,c) = value;
        }
      }
      Projections[name] = P;
    }
  }

  Eigen::Matrix<double,3,3> K = Projections.at("P0").block<3,3>(0,0);
  Eigen::Matrix<double,4,1> D;
  D.setZero();
  cv::Mat im = GetImage(0);
  int width  = im.cols;
  int height = im.rows;
  if( std::filesystem::exists(im1_path) ){
    // Stereo vision
    const Eigen::Matrix<double,3,4>& P1 = Projections.at("P1");
    auto t = K.inverse() * P1.block<3,1>(0,3);
    g2o::SE3Quat Trl;
    Trl.setTranslation(t);
    camera_ = new StereoCamera(K, D, K, D, Trl, width, height);
  }
  else if( std::filesystem::exists(depth_path) ){
    camera_ = new DepthCamera(K, D, width, height);
  }
  else{
    std::cout << "KittiDataset::KittiDataset() Failed to define camera type." << std::endl;
    exit(1);
  }
}

KittiDataset::~KittiDataset(){
  delete camera_;
}

cv::Mat KittiDataset::GetImage(int i, int flags) const{
  std::string fn = im0_filenames_.at(i);
  cv::Mat src = cv::imread(fn, flags);
  return src;
}

cv::Mat KittiDataset::GetRightImage(int i, int flags) const{
  std::string fn = im1_filenames_.at(i);
  cv::Mat src = cv::imread(fn, flags);
  return src;
}

cv::Mat KittiDataset::GetDepthImage(int i) const{
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
  return depth;
}


int KittiDataset::Size() const{
  return im0_filenames_.size();
}

const EigenMap<int, g2o::SE3Quat>& KittiDataset::GetTcws() const {
  return Tcws_;
}

const Camera* KittiDataset::GetCamera() const {
  return camera_;
}
