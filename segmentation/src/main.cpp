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

#include "../include/seg.h"
#include "../include/util.h"
#include "../include/seg.h"

#include "dataset.h"
#include "orb_extractor.h"
#include "frame.h"
#include "camera.h"
//#include "pipeline.h"
//#include "qmap_viewer.h"
//#include <QApplication>
//#include <QWidget>
#include "segslam.h"

int TestKitti(int argc, char** argv) {
  //Seq :"02",("13" "20");
  std::string seq(argv[2]);
  KittiDataset dataset(seq);
  const auto& Tcws = dataset.GetTcws();
  if(Tcws.empty()){
    std::cout << "Seq" << seq << " with no ground truth trajectory." << std::endl;
  }
  const auto& D = dataset.GetCamera()->GetD();
  std::cout << "Distortion = " << D.transpose() << std::endl;
  const StereoCamera* camera = dynamic_cast<const StereoCamera*>(dataset.GetCamera());
  assert(camera);
  Segmentor segmentor;

  bool stop = true;
  for(int i=0; i<dataset.Size(); i+=1){
    std::cout << "F# " << i << std::endl;
    const cv::Mat rgb   = dataset.GetImage(i, cv::IMREAD_COLOR);
    const cv::Mat rgb_r = dataset.GetRightImage(i, cv::IMREAD_COLOR);
    cv::Mat gray, gray_r;
    cv::cvtColor(rgb,gray,cv::COLOR_BGR2GRAY);
    cv::cvtColor(rgb_r,gray_r,cv::COLOR_BGR2GRAY);
    segmentor.Put(gray, gray_r, *camera, rgb);
    char c = cv::waitKey(stop?0:1);
    if(c == 'q')
      break;
    else if (c == 's')
      stop = !stop;
  }
  std::cout << "Done. The end of the sequence" << std::endl;
  return 1;
}

int TestWaymodataset(int argc, char** argv) {
  // 주어진 RGB+Dense Depthmap에서 
  const std::string dataset_path = GetPackageDir()+ "/../thirdparty/waymo-dataset/output/";
  const std::string seq(argv[2]);
  const std::string start = argc > 3 ? std::string(argv[3]) : "0";

  KittiDataset dataset(seq,dataset_path);
  const DepthCamera* camera = dynamic_cast<const DepthCamera*>(dataset.GetCamera());
  assert(camera);
  Segmentor segmentor;

  int nfeatures = 2000;
  float scale_factor = 1.2;
  int nlevels = 8;
  int initial_fast_th = 20;
  int min_fast_th = 7;
  ORB_SLAM2::ORBextractor extractor(nfeatures, scale_factor,
                                    nlevels, initial_fast_th, min_fast_th);
  seg::Pipeline pipeline(camera, &extractor);

  //std::cout << "Intrinsic = \n" << camera->GetK() << std::endl;
  const EigenMap<int, g2o::SE3Quat>& Tcws = dataset.GetTcws();
  bool stop = 0==std::stoi(start);
  for(int i=0; i<dataset.Size(); i+=1){
    const auto Twc = Tcws.at(i).inverse();
    //std::cout << "t=" << Twc.translation().transpose() << std::endl;

    const cv::Mat rgb   = dataset.GetImage(i,cv::IMREAD_UNCHANGED);
    const cv::Mat depth = dataset.GetDepthImage(i);
    cv::Mat gray;
    cv::cvtColor(rgb,gray,cv::COLOR_BGR2GRAY);
    const std::map<seg::Pth, ShapePtr>& shapes = segmentor.Put(gray, depth, *camera, rgb);
    pipeline.Put(gray, depth, shapes, rgb);

    char c = cv::waitKey(stop?0:100);
    if(c == 'q')
      break;
    else if (c == 's')
      stop = !stop;
  }
  return 1;
}

int main(int argc, char** argv){
  if(argc < 2){
    std::cout << "No arguments for dataset name." << std::endl;
    exit(-1);
  }
  const std::string dataset_name(argv[1]);
  if(dataset_name=="kitti")
      return TestKitti(argc, argv); // TODO depth update가 제대로 안된다
  else if(dataset_name=="waymo")
      return TestWaymodataset(argc, argv);

  return 1;
}
