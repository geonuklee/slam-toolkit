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
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <pybind11/embed.h>

void ComputeCacheOfKittiTrackingDataset(){
  pybind11::scoped_interpreter python; // 이 인스턴스가 파괴되면 인터프리터 종료.
  const std::string dataset_path = GetPackageDir()+ "/kitti_tracking_dataset/";
  const std::string dataset_type = "training";
  char buffer[24];
  for(size_t i =0; i < 21; i++){
    std::sprintf(buffer, "%04ld", i);
    const std::string seq(buffer);
    std::cout << "Do seq " << seq << std::endl;
    KittiTrackingDataset dataset(dataset_type, seq, dataset_path);
    //if(! dataset.EixstCachedDepthImages() )
      dataset.ComputeCacheImages();
  }
  return;
}

void TestKittiTrackingDataset(){
  const std::string dataset_path = GetPackageDir()+ "/kitti_tracking_dataset/";
  const std::string dataset_type = "training";
  const std::string seq = "0003";
  KittiTrackingDataset dataset(dataset_type, seq, dataset_path);
  for(int i =0; i < dataset.Size(); i++){
    cv::Mat rgb = dataset.GetImage(i);
    cv::Mat depth = dataset.GetDepthImage(i);
    cv::Mat instance_mask = dataset.GetInstanceMask(i);
    cv::Mat d_mask = dataset.GetDynamicMask(i);
    cv::imshow("dmask", 255*d_mask);
    cv::imshow("Ins mask", GetColoredLabel(instance_mask) );
    cv::imshow("rgb", rgb);
    cv::imshow("depth", .01*depth);
    char c = cv::waitKey(1);
    if(c=='q')
      break;
  }
  cv::waitKey();
  return;
}

#include "seg_viewer.h"
void TestPangolin(int argc, char** argv) {
  const std::string dataset_path = GetPackageDir()+ "/kitti_tracking_dataset/";
  const std::string dataset_type = "training";
  const std::string seq(argv[1]);
  KittiTrackingDataset dataset(dataset_type, seq, dataset_path);
  const EigenMap<int, g2o::SE3Quat>& gt_Tcws = dataset.GetTcws();
  const std::string config_fn = GetPackageDir()+"/config/kitti_tracking.yaml";
  SegViewer viewer(gt_Tcws, config_fn);
  bool stop = false;

  for(int i =0; i < dataset.Size(); i++){
    auto gt_Twc = gt_Tcws.at(i).inverse();
    cv::Mat rgb = dataset.GetImage(i);
    cv::Mat depth = dataset.GetDepthImage(i);
    cv::Mat instance_mask = dataset.GetInstanceMask(i);
    cv::Mat d_mask = dataset.GetDynamicMask(i);
    if(instance_mask.empty())
      instance_mask = cv::Mat::zeros(rgb.size(), CV_32SC1);
    if(d_mask.empty())
      d_mask = cv::Mat::zeros(rgb.size(), CV_8UC1);

    g2o::SE3Quat err;
    err.setTranslation(Eigen::Vector3d(1.,0., 0.2));
    g2o::SE3Quat est_Tcw = err *gt_Twc.inverse();
    viewer.SetCurrCamera(i, est_Tcw);

    cv::imshow("dmask", 255*d_mask);
    cv::imshow("Ins mask", GetColoredLabel(instance_mask) );
    cv::imshow("rgb", rgb);
    cv::imshow("depth", .01*depth);
    char c = cv::waitKey(stop?0:1);
    if(c=='q')
      break;
    else if(c=='s')
      stop = !stop;
  }
  bool req_exit = true;
  viewer.Join(req_exit); // close request 도 추가는 해야겠다.
  return;
}

int TestKittiTrackingOldSLAM(int argc, char** argv) {
  const std::string dataset_path = GetPackageDir()+ "/kitti_tracking_dataset/";
  const std::string dataset_type = "training";
  const std::string seq(argv[1]);
  KittiTrackingDataset dataset(dataset_type, seq, dataset_path);
  const EigenMap<int, g2o::SE3Quat>& gt_Tcws = dataset.GetTcws();
  const std::string config_fn = GetPackageDir()+"/config/kitti_tracking.yaml";
  SegViewer viewer(gt_Tcws, config_fn);
  bool stop = false;

  const auto& D = dataset.GetCamera()->GetD();
  const StereoCamera* camera = dynamic_cast<const StereoCamera*>(dataset.GetCamera());
  assert(camera);

  const auto Trl_ = camera->GetTrl();
  const float base_line = -Trl_.translation().x();
  const float fx = camera->GetK()(0,0);
  const float fy = camera->GetK()(1,1);
  const float min_disp = 1.;
  const float snyc_min_iou = .3;
  SEG::CvFeatureDescriptor extractor;
  std::shared_ptr<OutlineEdgeDetector> edge_detector( new OutlineEdgeDetectorWithSIMD );  // After   2.5 [milli sec]
  std::shared_ptr<Segmentor> segmentor( new SegmentorNew );                               // After  5~10 [milli sec] , with octave 2
  std::shared_ptr<ImageTrackerNew> img_tracker( new ImageTrackerNew);                     // Afte  10~11 [milli sec]
  OLD_SEG::Pipeline pipeline(camera, &extractor);                                             // After   ~50 [milli sec]
  std::string output_fn = std::string(PACKAGE_DIR)+"/output.txt";
  std::ofstream output_file(output_fn);
  if(! output_file.is_open() ){
    std::cout << "can't open " << output_fn << std::endl;
    throw -1;
  }

  for(int i=0; i<dataset.Size(); i+=1){
    std::cout << "F# " << i << std::endl;
    const cv::Mat rgb   = dataset.GetImage(i, cv::IMREAD_COLOR);
    const cv::Mat rgb_r = dataset.GetRightImage(i, cv::IMREAD_COLOR);
    cv::Mat gray, gray_r;
    cv::cvtColor(rgb, gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(rgb_r, gray_r, cv::COLOR_BGR2GRAY);
    cv::Mat depth = dataset.GetDepthImage(i);

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
    OLD_SEG::Frame* frame = nullptr;
    {
      cv::Mat modified_depth = depth.clone();
      modified_depth.setTo(0, depth > 40.);
      frame = pipeline.Put(gray, modified_depth, flow, synced_marker, marker_areas,
                           gradx, grady, valid_grad, rgb);
    }
    auto t7 = std::chrono::steady_clock::now();
    std::cout << "vslam piepline : " << std::setprecision(3) <<
      std::chrono::duration<float, std::milli>(t7-t6).count() << "[milli sec]" << std::endl;
    WriteKittiTrajectory(frame->GetTcq(0), output_file);
    pipeline.Visualize(rgb, gt_Tcws.empty()?nullptr:&gt_Tcws);
    viewer.SetCurrCamera(i, frame->GetTcq(0));
    char c = cv::waitKey(stop?0:1);
    if(c == 'q')
      break;
    else if (c == 's')
      stop = !stop;
  }
  bool req_exit = true;
  viewer.Join(req_exit); // close request 도 추가는 해야겠다.
  output_file.close();
  std::cout << "Done. The end of the sequence" << std::endl;
  std::cout << "Output on " << output_fn << std::endl;
  return 1;
}

int TestKittiTrackingNewSLAM(int argc, char** argv) {
  const std::string dataset_path = GetPackageDir()+ "/kitti_tracking_dataset/";
  const std::string dataset_type = "training";
  const std::string seq(argv[1]);
  KittiTrackingDataset dataset(dataset_type, seq, dataset_path);
  const EigenMap<int, g2o::SE3Quat>& gt_Tcws = dataset.GetTcws();
  const std::string config_fn = GetPackageDir()+"/config/kitti_tracking.yaml";
  SegViewer viewer(gt_Tcws, config_fn);
  bool stop = true;

  const auto& D = dataset.GetCamera()->GetD();
  const StereoCamera* camera = dynamic_cast<const StereoCamera*>(dataset.GetCamera());
  assert(camera);

  const auto Trl_ = camera->GetTrl();
  const float base_line = -Trl_.translation().x();
  const float fx = camera->GetK()(0,0);
  const float fy = camera->GetK()(1,1);
  const float min_disp = 1.;
  const float snyc_min_iou = .3;
  SEG::CvFeatureDescriptor extractor;
  std::shared_ptr<OutlineEdgeDetector> edge_detector( new OutlineEdgeDetectorWithSIMD );  // After   2.5 [milli sec]
  std::shared_ptr<Segmentor> segmentor( new SegmentorNew );                               // After  5~10 [milli sec] , with octave 2
  std::shared_ptr<ImageTrackerNew> img_tracker( new ImageTrackerNew);                     // Afte  10~11 [milli sec]
  NEW_SEG::Pipeline pipeline(camera, &extractor);                                             // After   ~50 [milli sec]
  std::string output_fn = std::string(PACKAGE_DIR)+"/output.txt";

  char c = 0;
  g2o::SE3Quat Tc0w;
  Tc0w.setTranslation(Eigen::Vector3d(0.,0.,1.));
  for(int i=0; i<dataset.Size(); i+=1){
    std::cout << "F# " << i << std::endl;
    const cv::Mat rgb   = dataset.GetImage(i, cv::IMREAD_COLOR);
    const cv::Mat rgb_r = dataset.GetRightImage(i, cv::IMREAD_COLOR);
    cv::Mat gray, gray_r;
    cv::cvtColor(rgb, gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(rgb_r, gray_r, cv::COLOR_BGR2GRAY);
    cv::Mat depth = dataset.GetDepthImage(i);
    edge_detector->PutDepth(depth, fx, fy);
    cv::Mat gradx = edge_detector->GetGradx();
    cv::Mat grady = edge_detector->GetGrady();
    cv::Mat outline_edges = edge_detector->GetOutline();
    cv::Mat valid_mask = edge_detector->GetValidMask();
    cv::Mat valid_grad = valid_mask;
    segmentor->Put(outline_edges, valid_mask);
    cv::Mat unsync_marker = segmentor->GetMarker();
    img_tracker->Put(gray, unsync_marker, snyc_min_iou);
    const std::vector<cv::Mat>& flow = img_tracker->GetFlow();
    //if(!flow.empty()) cv::imshow("flow",VisualizeFlow(flow));
    cv::Mat synced_marker = img_tracker->GetSyncedMarker();
    const std::map<int,size_t>& marker_areas = img_tracker->GetMarkerAreas();
    cv::Mat modified_depth = depth.clone();
    //modified_depth.setTo(0, depth > 40.);
    NEW_SEG::Frame* frame = pipeline.Put(gray, modified_depth, flow, synced_marker, marker_areas,
                                         gradx, grady, valid_grad, rgb);
    g2o::SE3Quat Tcw = frame->GetTcq(0);
    /* if( (Tc0w * Tcw.inverse() ).translation().z() < 0.) { // 계산실패.
      stop=true;
      cv::waitKey(0);
    } */
    Tc0w = Tcw;
    pipeline.Visualize(rgb);
    viewer.SetCurrCamera(i, Tcw);
    viewer.SetMappoints(frame->Get3dMappoints());
    //cv::imshow("marker", GetColoredLabel(synced_marker) );
    c = cv::waitKey(stop?0:1);
    if(c == 'q')
      break;
    else if (c == 's')
      stop = !stop;
  }
  bool req_exit = c=='q';
  viewer.Join(req_exit);
  return 1;
}

int main(int argc, char** argv){
  //ComputeCacheOfKittiTrackingDataset();
  //TestKittiTrackingDataset();
  //TestPangolin(argc, argv);
  //TestKittiTrackingOldSLAM(argc, argv);
  TestKittiTrackingNewSLAM(argc, argv);
  return 1;
}

