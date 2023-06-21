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

ORB_SLAM2::ORBextractor* KittiDataset::GetExtractor() const {
  return extractor_;
}

KittiDataset::KittiDataset(std::string seq)
  : extractor_(nullptr)
{
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
  std::string dataset_path = GetPackageDir() + "/kitti_odometry_dataset";
  std::string im0_path = dataset_path+"/sequences/"+seq+"/image_0";
  std::string im1_path = dataset_path+"/sequences/"+seq+"/image_1";
  boost::filesystem::directory_iterator it{im0_path}, end;
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
    im1_filenames_[i] = im1_path+"/" + it->path().filename().string();
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

  Eigen::Matrix<double,3,3> K;
  g2o::SE3Quat Trl;
  K << 7.188560000000e+02, 0., 6.071928e+02,
      0., 7.188560000000e+02, 1.852157000000e+02,
      0., 0., 1.;

  Eigen::Matrix<double,4,1> D;
  D.setZero();

  Eigen::Matrix<double,3,4> P1; // KR | Kt
  P1 << 7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02, -3.861448000000e+02,
     0.000000000000e+00, 7.188560000000e+02, 1.852157000000e+02, 0.000000000000e+00,
     0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 0.000000000000e+00;
  auto t = K.inverse() * P1.block<3,1>(0,3);
  Trl.setTranslation(t);

  int width = 1241;
  int height = 376;
  camera_ = new StereoCamera(K, D, K, D, Trl, width, height);
}

KittiDataset::~KittiDataset(){
  if(extractor_)
    delete extractor_;
  delete camera_;
}

cv::Mat KittiDataset::GetImage(int i) const{
  std::string fn = im0_filenames_.at(i);
  cv::Mat src = cv::imread(fn, cv::IMREAD_GRAYSCALE);
  return src;
}

cv::Mat KittiDataset::GetRightImage(int i) const{
  std::string fn = im1_filenames_.at(i);
  cv::Mat src = cv::imread(fn, cv::IMREAD_GRAYSCALE);
  return src;
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
