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

//#include "dataset.h"
#include "seg_dataset.h"
#include "orb_extractor.h"
#include "frame.h"
#include "camera.h"
#include "segslam.h"
#include <exception>
#include <g2o/types/slam3d/se3quat.h>
#include <filesystem>

#include <iomanip>
#include <memory>
#include <opencv2/imgcodecs.hpp>
#include <pybind11/embed.h>
#include "hitnet.h"

/*
// https://stackoverflow.com/questions/17735863/opencv-save-cv-32fc1-images
bool writeRawImage(const cv::Mat& image, const std::string& filename) {
  std::ofstream file;
    file.open (filename, std::ios::out|std::ios::binary);
    if (!file.is_open())
        return false;
    file.write(reinterpret_cast<const char *>(&image.rows), sizeof(int));
    file.write(reinterpret_cast<const char *>(&image.cols), sizeof(int));
    const int depth = image.depth();
    const int type  = image.type();
    const int channels = image.channels();
    file.write(reinterpret_cast<const char *>(&depth), sizeof(depth));
    file.write(reinterpret_cast<const char *>(&type), sizeof(type));
    file.write(reinterpret_cast<const char *>(&channels), sizeof(channels));
    int sizeInBytes = image.step[0] * image.rows;
    file.write(reinterpret_cast<const char *>(&sizeInBytes), sizeof(int));
    file.write(reinterpret_cast<const char *>(image.data), sizeInBytes);
    file.close();
    return true;
}

cv::Mat readRawImage(const std::string& filename) {
  cv::Mat image;
  int rows, cols, data, depth, type, channels;
  std::ifstream file (filename, std::ios::in|std::ios::binary);
  if (!file.is_open())
    return image;
  try {
    file.read(reinterpret_cast<char *>(&rows), sizeof(rows));
    file.read(reinterpret_cast<char *>(&cols), sizeof(cols));
    file.read(reinterpret_cast<char *>(&depth), sizeof(depth));
    file.read(reinterpret_cast<char *>(&type), sizeof(type));
    file.read(reinterpret_cast<char *>(&channels), sizeof(channels));
    file.read(reinterpret_cast<char *>(&data), sizeof(data));
    image = cv::Mat(rows, cols, type);
    file.read(reinterpret_cast<char *>(image.data), data);
  } catch (...) {
    file.close();
    return image;
  }

  file.close();
  return image;
}
*/


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

#if 0
#define USE_DEPTHFILE
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
  const float snyc_min_iou = .3;

  pybind11::scoped_interpreter python; // 이 인스턴스가 파괴되면 인터프리터 종료.
#ifdef USE_DEPTHFILE
  std::string images_dir = "../output_images/";
#else
  HITNetStereoMatching hitnet;
#endif

  seg::CvFeatureDescriptor extractor;
  /* Comparison after computation time optimization */
  //std::shared_ptr<OutlineEdgeDetector> edge_detector( new OutlineEdgeDetectorWithoutSIMD ); // Before   76 [milli sec]
  //std::shared_ptr<Segmentor> segmentor( new SegmentorOld );                                 // Before  110 [milli sec]
  //std::shared_ptr<ImageTracker> img_tracker( new ImageTrackerOld);                          // Before  250 [milli sec]
  //seg::Pipeline pipeline(camera, &extractor);                                               // Before  150 [milli sec]

  std::shared_ptr<OutlineEdgeDetector> edge_detector( new OutlineEdgeDetectorWithSIMD );  // After   2.5 [milli sec]
  std::shared_ptr<Segmentor> segmentor( new SegmentorNew );                               // After  5~10 [milli sec] , with octave 2
  std::shared_ptr<ImageTrackerNew> img_tracker( new ImageTrackerNew);                     // Afte  10~11 [milli sec]
  seg::Pipeline pipeline(camera, &extractor);                                             // After   ~50 [milli sec]

  std::string output_fn = std::string(PACKAGE_DIR)+"/output.txt";
  std::ofstream output_file(output_fn);
  if(! output_file.is_open() ){
    std::cout << "can't open " << output_fn << std::endl;
    throw -1;
  }

  bool visualize_segment = true;
  bool stop = true;
  for(int i=0; i<dataset.Size(); i+=1){
    std::cout << "F# " << i << std::endl;
    const cv::Mat rgb   = dataset.GetImage(i, cv::IMREAD_COLOR);
    const cv::Mat rgb_r = dataset.GetRightImage(i, cv::IMREAD_COLOR);
    cv::Mat gray, gray_r;
    cv::cvtColor(rgb, gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(rgb_r, gray_r, cv::COLOR_BGR2GRAY);
#ifdef USE_DEPTHFILE
    std::string fn = images_dir+"depth"+std::to_string(i)+".raw";
    cv::Mat depth = readRawImage(fn);
    if(depth.empty()){
      std::cout << "end of depth file, " << fn << std::endl;
      break;
    }
#else
    cv::Mat disp  = hitnet.GetDisparity(gray, gray_r);
    cv::Mat depth = Disparit2Depth(disp,base_line,fx, min_disp);
#endif

    auto t0 = std::chrono::steady_clock::now();
    edge_detector->PutDepth(depth, fx, fy);
    auto t1 = std::chrono::steady_clock::now();

    cv::Mat gradx = edge_detector->GetGradx();
    cv::Mat grady = edge_detector->GetGrady();
    cv::Mat outline_edges = edge_detector->GetOutline();
    cv::Mat valid_mask = edge_detector->GetValidMask();
    cv::Mat valid_grad = valid_mask;

    std::cout << "outline edge : " << std::setprecision(3) <<
      std::chrono::duration<float, std::milli>(t1-t0).count() << "[milli sec]" << std::endl;

    auto t2 = std::chrono::steady_clock::now();
    segmentor->Put(outline_edges, valid_mask);
    cv::Mat unsync_marker = segmentor->GetMarker();
    auto t3 = std::chrono::steady_clock::now();
    std::cout << "segment : " << std::setprecision(3) <<
      std::chrono::duration<float, std::milli>(t3-t2).count() << "[milli sec]" << std::endl;

    auto t4 = std::chrono::steady_clock::now();
    img_tracker->Put(gray, unsync_marker, snyc_min_iou);
    const std::vector<cv::Mat>& flow = img_tracker->GetFlow();
    cv::Mat synced_marker = img_tracker->GetSyncedMarker();
    const std::map<int,size_t>& marker_areas = img_tracker->GetMarkerAreas();
    auto t5 = std::chrono::steady_clock::now();
    std::cout << "track image : " << std::setprecision(3) <<
      std::chrono::duration<float, std::milli>(t5-t4).count() << "[milli sec]" << std::endl;
    auto t6 = std::chrono::steady_clock::now();
    seg::Frame* frame = nullptr;
    try {
      cv::Mat modified_depth = depth.clone();
      modified_depth.setTo(0, depth > 40.);
      frame = pipeline.Put(gray, modified_depth, flow, synced_marker, marker_areas,
                           gradx, grady, valid_grad, rgb);
    }
    catch(const std::exception& e) {
      if(std::string(e.what())=="termination")
        exit(1);
    }
    auto t7 = std::chrono::steady_clock::now();
    std::cout << "vslam piepline : " << std::setprecision(3) <<
      std::chrono::duration<float, std::milli>(t7-t6).count() << "[milli sec]" << std::endl;
    WriteKittiTrajectory(frame->GetTcq(0), output_file);
    pipeline.Visualize(rgb, Tcws.empty()?nullptr:&Tcws);

    /*
    cv::imshow("rgb", rgb);
    cv::imshow("outline", 255*outline_edges);
    cv::imshow("segment", GetColoredLabel(synced_marker) );
    */

    // TODO vSLAM 과정의 visualization
#ifndef USE_DEPTHFILE
    cv::imshow("depth", .01*depth);
    cv::imwrite("../output_images/gray"+std::to_string(i)+".png",gray);
    cv::imwrite("../output_images/outline"+std::to_string(i)+".png",outline_edges);
    writeRawImage(depth, "../output_images/depth"+std::to_string(i)+".raw");
#endif

    char c = cv::waitKey(stop?0:1);
    if(c == 'q')
      break;
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
#endif

void TestKittiTrackingDataset(){
  const std::string dataset_path = GetPackageDir()+ "/kitti_tracking_dataset/";
  const std::string dataset_type = "training";
  const std::string seq = "0000";
  KittiTrackingDataset dataset(dataset_type, seq, dataset_path);
  /* TODO
  * [ ] trajectory 를 mat plot lib으로 그리기.
  * [ ] segmented instance를 python 호출해서 불러오기.
  */
  if(! dataset.EixstCachedDepthImages() )
    dataset.ComputeCacheImages();

  return;
}

int main(int argc, char** argv){
  TestKittiTrackingDataset();
  return 1;
}

