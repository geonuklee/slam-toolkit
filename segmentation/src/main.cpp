/*
Copyright (c) 2023 Geonuk Lee

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

#include "dataset.h"
#include "orb_extractor.h"
#include "frame.h"
#include "camera.h"
//#include "pipeline.h"
//#include "qmap_viewer.h"
//#include <QApplication>
//#include <QWidget>
#include "segslam.h"
#include <exception>
#include <g2o/types/slam3d/se3quat.h>
#include <filesystem>

#include <iomanip>
#include <memory>
#include <opencv2/imgcodecs.hpp>
#include <pybind11/embed.h>
#include "hitnet.h"

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

  const auto Trl_ = camera->GetTrl();
  const float base_line = -Trl_.translation().x();
  const float fx = camera->GetK()(0,0);
  const float fy = camera->GetK()(1,1);
  const float min_disp = 1.;

  pybind11::scoped_interpreter python; // 이 인스턴스가 파괴되면 인터프리터 종료.
  HITNetStereoMatching hitnet;

  //std::shared_ptr<OutlineEdgeDetector> edge_detector( new OutlineEdgeDetectorWithoutSIMD ); // Before   76 [milli sec]
  std::shared_ptr<OutlineEdgeDetector> edge_detector( new OutlineEdgeDetectorWithSIMD );      // After   2.5 [milli sec]
  //std::shared_ptr<Segmentor> segmentor( new SegmentorOld );                                 // Before  110 [milli sec]
  std::shared_ptr<Segmentor> segmentor( new SegmentorNew );                                   // After  5~10 [milli sec]
  std::shared_ptr<ShapeTracker> shape_tracker( new ShapeTrackerOld);                          // Before  250 [milli sec]

  seg::CvFeatureDescriptor extractor;
  seg::Pipeline pipeline(camera, &extractor);                                                 //  150 [milli sec]

  std::string output_fn = std::string(PACKAGE_DIR)+"/output.txt";
  std::ofstream output_file(output_fn);
  if(! output_file.is_open() ){
    std::cout << "can't open " << output_fn << std::endl;
    throw -1;
  }

  bool visualize_segment = true;
  bool stop = false;
  for(int i=0; i<dataset.Size(); i+=1){
    std::cout << "F# " << i << std::endl;
    const cv::Mat rgb   = dataset.GetImage(i, cv::IMREAD_COLOR);
    const cv::Mat rgb_r = dataset.GetRightImage(i, cv::IMREAD_COLOR);
    cv::Mat gray, gray_r;
    cv::cvtColor(rgb, gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(rgb_r, gray_r, cv::COLOR_BGR2GRAY);
    cv::Mat disp  = hitnet.GetDisparity(gray, gray_r);
    cv::Mat depth = Disparit2Depth(disp,base_line,fx, min_disp);

    auto t0 = std::chrono::steady_clock::now();
    edge_detector->PutDepth(depth, fx, fy);
    auto t1 = std::chrono::steady_clock::now();

    cv::Mat gradx = edge_detector->GetGradx();
    cv::Mat grady = edge_detector->GetGrady();
    cv::Mat outline_edges = edge_detector->GetOutline();
    cv::Mat valid_mask = edge_detector->GetValidMask();
    cv::Mat valid_grad = valid_mask;

    auto t2 = std::chrono::steady_clock::now();
    segmentor->Put(outline_edges, valid_mask);
    cv::Mat marker = segmentor->GetMarker();
    auto t3 = std::chrono::steady_clock::now();
    std::cout << "outline edge : " << std::setprecision(3) <<
      std::chrono::duration<float, std::milli>(t1-t0).count() << "[milli sec]" << std::endl;

    std::cout << "segment : " << std::setprecision(3) <<
      std::chrono::duration<float, std::milli>(t3-t2).count() << "[milli sec]" << std::endl;

    //depth.setTo(0., depth > 50.);
    auto t4 = std::chrono::steady_clock::now();
    const std::map<seg::Pth, ShapePtr>& shapes = shape_tracker->Put(gray, marker);
    auto t5 = std::chrono::steady_clock::now();
    std::cout << "track shapes : " << std::setprecision(3) <<
      std::chrono::duration<float, std::milli>(t5-t4).count() << "[milli sec]" << std::endl;

    cv::Mat flow0 = shape_tracker->GetFlow0();

    auto t6 = std::chrono::steady_clock::now();
    seg::Frame* frame = nullptr;
    try {
      frame = pipeline.Put(gray, depth, flow0, shapes, gradx, grady, valid_grad, rgb, Tcws.empty()?nullptr:&Tcws);
    }
    catch(const std::exception& e) {
      if(std::string(e.what())=="termination")
        exit(1);
    }
    auto t7 = std::chrono::steady_clock::now();
    std::cout << "vslam piepline : " << std::setprecision(3) <<
      std::chrono::duration<float, std::milli>(t7-t6).count() << "[milli sec]" << std::endl;

    WriteKittiTrajectory(frame->GetTcq(0), output_file);

    cv::imshow("rgb", rgb);
    cv::imshow("outline", 255*outline_edges);
    cv::imshow("depth", .01*depth);
    cv::imshow("segment", GetColoredLabel(marker) );
    char c = cv::waitKey(stop?0:1);
    if(c == 'q')
      break;
    else if (c == 'f'){
      /*
      cv::imwrite("outline.bmp", outline_edges);
      cv::imwrite("rgb.bmp", rgb);
      */
    }
    else if (c == 's')
      stop = !stop;
  }
  output_file.close();
  std::cout << "Done. The end of the sequence" << std::endl;
  std::cout << "Output on " << output_fn << std::endl;
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
  //Segmentor segmentor;

  seg::CvFeatureDescriptor extractor;
  seg::Pipeline pipeline(camera, &extractor);
  bool visualize_segment = true;

  //std::cout << "Intrinsic = \n" << camera->GetK() << std::endl;
  const EigenMap<int, g2o::SE3Quat> Tcws = dataset.GetTcws();
  bool stop = 0==std::stoi(start);
  for(int i=0; i<dataset.Size(); i+=1){
    //const auto Twc = Tcws.at(i).inverse();
    //std::cout << i << ", gt t=" << Twc.translation().transpose() << std::endl;
    const cv::Mat rgb   = dataset.GetImage(i,cv::IMREAD_UNCHANGED);
    const cv::Mat depth = dataset.GetDepthImage(i);
    cv::Mat gray, flow0, gradx, grady, valid_grad;
    cv::cvtColor(rgb,gray,cv::COLOR_BGR2GRAY);
    //const std::map<seg::Pth, ShapePtr>& shapes = segmentor.Put(gray, depth, camera, visualize_segment ? rgb : cv::Mat(), flow0, gradx, grady, valid_grad);
    //pipeline.Put(gray, depth, flow0, shapes, gradx, grady, valid_grad, rgb, &Tcws);
    //cv::imshow("depth", 0.01*depth);
    /*
    if(i<1)
      continue;
    char c = cv::waitKey(stop?0:1);
    if(c == 'q')
      break;
    else if (c == 's')
      stop = !stop;
    */
  }
  std::cout << "Done. The end of the dataset." << std::endl;
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
