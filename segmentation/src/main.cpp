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
#include "floorseg.h"
#include <exception>
#include <g2o/types/slam3d/se3quat.h>
#include <filesystem>

#include <iomanip>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
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

void DrawFigureForPaper(cv::Mat rgb, cv::Mat depth, cv::Mat outline_edges, cv::Mat synced_marker,
                        const NEW_SEG::Frame* frame,
                        std::string output_dir){
  cv::imshow("rgb", rgb);
  cv::Mat ndepth; {
    double minDepth, maxDepth;
    cv::minMaxIdx(depth, &minDepth, &maxDepth);
    depth.convertTo(ndepth, CV_8UC1, 255.0 / (maxDepth - minDepth), -minDepth * 255.0 / (maxDepth - minDepth));
  }

  cv::imshow("outline", outline_edges*255);
  cv::Mat vis_marker = GetColoredLabel(synced_marker);
  //vis_marker.setTo(CV_RGB(0,0,0), GetBoundary(synced_marker));
  cv::imshow("segmentation", vis_marker );
  cv::Mat vis_points = cv::Mat::zeros(rgb.rows, rgb.cols, rgb.type());
  const auto& keypoints = frame->GetKeypoints();
  const auto& mappoints = frame->GetMappoints();
  std::map<Qth, std::set<NEW_SEG::Instance*> > instances;
  for(int i = 0; i < keypoints.size(); i++){
    NEW_SEG::Mappoint* mp = mappoints.at(i);
    if(!mp)
      continue;
    NEW_SEG::Instance* ins = mp->GetLatestInstance();
    const auto& color = colors.at(ins->GetId() % colors.size() );
    const auto& kpt = keypoints.at(i);
    cv::circle(vis_points, kpt.pt, 3, color, -1);
    if(ins->GetQth() != 0)
      instances[ins->GetQth()].insert(ins);
  }
  cv::Mat vis_dynamics; {
    vis_dynamics = rgb.clone();
    for(auto it : instances){
      if(it.first == 0)
        continue;
      for(NEW_SEG::Instance* ins : it.second)
        vis_dynamics.setTo(CV_RGB(255,0,0),synced_marker==ins->GetId());
    }
    cv::addWeighted(vis_dynamics, 0.7, rgb, 0.3, 1., vis_dynamics);
  }
  cv::Mat boundary = GetBoundary(synced_marker);
  vis_points.setTo(CV_RGB(255,255,255),boundary);
  cv::imshow("labeled mappoints", vis_points);
  cv::imshow("dynamics", vis_dynamics);
  cv::imwrite(output_dir+"/ex_rgb.png", rgb);
  cv::imwrite(output_dir+"/ex_outline.png", outline_edges*255);
  cv::imwrite(output_dir+"/ex_segment.png", vis_marker);
  cv::imwrite(output_dir+"/ex_labeledpoints.png", vis_points);
  cv::imwrite(output_dir+"/ex_dynamics.png", vis_dynamics);
  cv::imshow("ndepth", ndepth);
  cv::imwrite(output_dir+"/ex_depth.png", ndepth);
  return;
}

#include "occlusion_seg.h"
int TestConcaveEdges(int argc, char** argv){
  if(argc < 3){
    std::cout << "Need 3 argc" << std::endl;
    std::cout << argc << std::endl;
    std::cout << argv[2] << std::endl;
    return -1;
  }
  int offset = 0;
  if(argc == 4)
    offset = std::stoi(std::string(argv[3]));
  const std::string dataset_path = GetPackageDir()+ "/kitti_tracking_dataset/";
  const std::string dataset_type = "training";
  const std::string seq(argv[1]);
  KittiTrackingDataset dataset(dataset_type, seq, dataset_path);
  const StereoCamera* camera = dynamic_cast<const StereoCamera*>(dataset.GetCamera());
  assert(camera);
  const float fx = camera->GetK()(0,0);
  const float fy = camera->GetK()(1,1);
  const float snyc_min_iou = .5;

  //auto edge_detector = new OutlineEdgeDetectorWithSizelimit();
  auto segmentor = new TotalSegmentor(fx,fy);
  auto img_tracker = new ImageTrackerNew();

  int i = 0; {
    for(i=0; i<offset; i+=1){
    }
  }
  bool stop = true;
  for(; i < dataset.Size(); i++){
    cv::Mat rgb = dataset.GetImage(i);
    cv::Mat depth = dataset.GetDepthImage(i);
    cv::imshow("rgb", rgb);
    segmentor->Put(depth);
    cv::Mat gray;
    cv::cvtColor(rgb, gray, cv::COLOR_BGR2GRAY);
    char c = cv::waitKey(stop?0:1);
    if(c=='q')
      break;
    else if (c == 's')
      stop = !stop;
  }
  cv::waitKey();
  return 0;
}


#include "seg_viewer.h"
int TestKittiTrackingNewSLAM(int argc, char** argv) {
  if(argc < 3){
    std::cout << "Need 3 argc" << std::endl;
    std::cout << argc << std::endl;
    std::cout << argv[2] << std::endl;
    return -1;
  }
  int offset = 0;
  if(argc == 4)
    offset = std::stoi(std::string(argv[3]));
  const std::string dataset_path = GetPackageDir()+ "/kitti_tracking_dataset/";
  const std::string dataset_type = "training";
  const std::string seq(argv[1]);
  KittiTrackingDataset dataset(dataset_type, seq, dataset_path);
  const EigenMap<int, g2o::SE3Quat>& gt_Tcws = dataset.GetTcws();
  {
    g2o::SE3Quat Twc0 = gt_Tcws.begin()->second.inverse();
    g2o::SE3Quat Twcf = gt_Tcws.rbegin()->second.inverse();
    double translation = (Twcf.translation()-Twc0.translation()).norm();
    if(translation < 10){
      std::cout << "Pass seq for too short translation" << std::endl;
      return 0;
    }
  }
  cv::Size dst_size;{
    //const auto& rgb = dataset.GetImage(0);
    //dst_size.width = rgb.cols;
    //dst_size.height = 2*rgb.rows;
    dst_size = cv::Size(900,900);
  }

  const std::string config_fn = GetPackageDir()+"/config/kitti_tracking.yaml";
  SegViewer viewer(gt_Tcws, config_fn, dst_size, "KITTI tracking seq "+seq);
  const auto& D = dataset.GetCamera()->GetD();
  const StereoCamera* camera = dynamic_cast<const StereoCamera*>(dataset.GetCamera());
  assert(camera);
  const auto Trl_ = camera->GetTrl();
  const float base_line = -Trl_.translation().x();
  const float fx = camera->GetK()(0,0);
  const float fy = camera->GetK()(1,1);
  const float cx = camera->GetK()(0,2);
  const float cy = camera->GetK()(1,2);
  const float min_disp = 1.;
  const float snyc_min_iou = .5;
  std::shared_ptr<OutlineEdgeDetector> edge_detector( new OutlineEdgeDetectorWithSIMD );  // After   2.5 [milli sec]
#if 0
  SegmentorNew segmentor;                                                                    // After  5~10 [milli sec] , with octave 2
#else
  Floorseg segmentor(fx,fy,cx,cy,base_line);
#endif

  std::shared_ptr<ImageTrackerNew> img_tracker( new ImageTrackerNew);                     // Afte  10~11 [milli sec]
  NEW_SEG::Pipeline pipeline(camera);                                                     // After   ~50 [milli sec]

  std::string output_dir = std::string(PACKAGE_DIR)+std::string("/")+std::string(argv[2]);
  bool stop = std::string(argv[2]).find("output_batch") > 0;
  bool req_exit = true;

  if(! std::filesystem::exists(output_dir) )
    std::filesystem::create_directories(output_dir);
  NEW_SEG::EvalWriter eval_writer(dataset_type, seq, output_dir);

  cv::Mat empty_dst = cv::Mat::zeros(dst_size.height, dst_size.width, CV_8UC3);
  g2o::SE3Quat TCw;
  int i = 0; {
    EigenMap<Jth, g2o::SE3Quat> updated_Tcws;
    for(i=0; i<offset; i+=1){
      TCw = gt_Tcws.at(i);
      updated_Tcws[i] = TCw;
    }
    if(!updated_Tcws.empty())
      viewer.SetCurrCamera(i-1, updated_Tcws, empty_dst);
  }
  for(; i<dataset.Size(); i+=1){
    const cv::Mat rgb   = dataset.GetImage(i, cv::IMREAD_COLOR);
    const cv::Mat rgb_r = dataset.GetRightImage(i, cv::IMREAD_COLOR);
    double frame_second = dataset.GetSecond(i);
    cv::Mat gray;
    cv::cvtColor(rgb, gray, cv::COLOR_BGR2GRAY);
    cv::Mat depth = dataset.GetDepthImage(i);

#if 0
    edge_detector->PutDepth(depth, fx, fy);
    cv::Mat gradx = edge_detector->GetGradx();
    cv::Mat grady = edge_detector->GetGrady();
    cv::Mat outline_edges = edge_detector->GetOutline();
    cv::Mat valid_mask = edge_detector->GetValidMask();
    cv::Mat valid_grad = valid_mask;

    cv::Mat invalid_depthmask = GetInvalidDepthMask(gray,30.); {
      cv::Mat far_invalid_depth;
      cv::bitwise_and(invalid_depthmask, depth > 50., far_invalid_depth);
      outline_edges.setTo(1, far_invalid_depth);
      //outline_edges.setTo(1, invalid_depthmask);
    }
    segmentor.Put(outline_edges); // SegmentorNew 는 outline_edge를 기반으로 segmentation.
    std::list<int> fixed_instances;
#else
    segmentor.Put(depth);         // Floorseg 는 depth를 기반으로 segmentation.
    cv::Mat gradx, grady, valid_grad;
    std::list<int> fixed_instances = {1};
#endif
    cv::Mat unsync_marker = segmentor.GetMarker();
    img_tracker->Put(gray, unsync_marker, snyc_min_iou);
    const std::vector<cv::Mat>& flow = img_tracker->GetFlow();
    cv::Mat synced_marker = img_tracker->GetSyncedMarker();
    const std::map<int,size_t>& marker_areas = img_tracker->GetMarkerAreas();
    //depth.setTo(0., invalid_depthmask); // SLAM pose tracking을 망친다.
    //cv::imshow("depth", .01*depth);
    //cv::imshow("gradaxy", .1* (cv::abs(gradx)+cv::abs(grady)) );
    NEW_SEG::Frame* frame = pipeline.Put(gray, depth, flow, synced_marker, marker_areas,
                                         fixed_instances,
                                         gradx, grady, valid_grad, frame_second, rgb);
    img_tracker->ChangeSyncedMarker(synced_marker);
    g2o::SE3Quat Tcw = frame->GetTcq(0) * TCw;
    bool is_kf = frame->IsKeyframe();
    cv::Mat dst; {
      cv::Mat dynamic_mask = dataset.GetDynamicMask(i);
      pipeline.Visualize(rgb, dynamic_mask, dst);
    }
    EigenMap<Jth, g2o::SE3Quat> updated_Tcws;
    for(auto it : pipeline.GetUpdatedTcqs()){
      updated_Tcws[it.first+offset] = it.second*TCw;
    }
    viewer.SetCurrCamera(i, updated_Tcws, dst);
    cv::Mat gt_insmask = dataset.GetInstanceMask(i);
    cv::Mat gt_dmask = dataset.GetDynamicMask(i);
    NEW_SEG::RigidGroup* rig = pipeline.GetRigidGroup(0);
    eval_writer.Write(frame, rig, synced_marker, gt_insmask, gt_dmask);
    //DrawFigureForPaper(rgb, depth, outline_edges, synced_marker, frame, output_dir);
    if(viewer.IsShutDowned())
      break;
    char c = cv::waitKey(stop?0:1);
    if(c == 'q'){
      req_exit = true;
      break;
    }
    else if (c == 's')
      stop = !stop;
  }
  eval_writer.WriteInstances(pipeline.GetPthRemoved2Replacing());
  viewer.Join(req_exit);
  return 1;
}

int TestFloorseg(int argc, char** argv){
  if(argc < 3){
    std::cout << "Need 3 argc" << std::endl;
    std::cout << argc << std::endl;
    std::cout << argv[2] << std::endl;
    return 1;
  }
  int offset = 0;
  if(argc == 4)
    offset = std::stoi(std::string(argv[3]));
  const std::string dataset_path = GetPackageDir()+ "/kitti_tracking_dataset/";
  const std::string dataset_type = "training";
  const std::string seq(argv[1]);
  KittiTrackingDataset dataset(dataset_type, seq, dataset_path);
  const StereoCamera* camera = dynamic_cast<const StereoCamera*>(dataset.GetCamera());
  assert(camera);
  const float fx = camera->GetK()(0,0);
  const float fy = camera->GetK()(1,1);
  const float cx = camera->GetK()(0,2);
  const float cy = camera->GetK()(1,2);
  const auto Trl_ = camera->GetTrl();
  const float base_line = -Trl_.translation().x();
  const float snyc_min_iou = .5;
  Floorseg floorseg(fx,fy,cx,cy,base_line);

  int i = 0; {
    for(i=0; i<offset; i+=1){
    }
  }
  bool stop = true;
  //for(; i < dataset.Size(); i++){
  while(true) {
    cv::Mat rgb = dataset.GetImage(i);
    cv::Mat depth = dataset.GetDepthImage(i);
    int c;
    do {
#if 1
      floorseg.Put(depth, rgb);
#else
      cv::Mat marker = floorseg.Put(depth); {
        cv::Mat dst = GetColoredLabel(marker,true);
        dst.setTo(CV_RGB(0,0,0), GetBoundary(marker,4));
        cv::addWeighted(rgb, .3, dst, .5, 1., dst);
        cv::imshow("dst", dst);
      }
#endif
      c = cv::waitKey(1);
      if(c < 0)
        continue;
      else if(c=='q')
        break;
      else if (c == 's')
        stop = !stop;
      else{
        i++;
        break;
      }
    }
    while(stop);
    i++;
    if(c == 'q')
      break;
  }
  return 0;
}

int main(int argc, char** argv){
  // TODO 하드코딩 제거.
  //ComputeCacheOfKittiTrackingDataset(); // TODO 필요한지여부 체크.
  //return 0;
  //TestKittiTrackingDataset();
  //TestPangolin(argc, argv);
  return TestKittiTrackingNewSLAM(argc, argv);
  //return TestConcaveEdges(argc, argv); // TotalSegmentor 결과가 더 나쁘다.
  //return TestFloorseg(argc, argv);
}

