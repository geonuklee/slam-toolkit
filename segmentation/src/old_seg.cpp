#include <opencv2/core/types.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudastereo.hpp>
// #include <opencv2/cudafilters.hpp>

#include "../include/seg.h"
#include "../include/util.h"

void Seg::Put(cv::Mat rgb, cv::Mat rgb_r, const StereoCamera& camera) {
  if(camera.GetD().norm() > 1e-5){
    std::cerr << "Not support distorted image. Put rectified image" << std::endl;
    exit(1);
  }

  cv::Mat gray;
  cv::cvtColor(rgb,gray,cv::COLOR_BGR2GRAY);
  cv::cuda::GpuMat g_gray;
  g_gray.upload(gray);

  cv::Mat gray_r;
  cv::cvtColor(rgb_r,gray_r,cv::COLOR_BGR2GRAY);
  cv::cuda::GpuMat g_gray_r;
  g_gray_r.upload(gray_r);
  cv::Mat disparity = GetDisparity(g_gray,g_gray_r);

  const auto Trl_ = camera.GetTrl();
  const float base_line = -Trl_.translation().x();
  const float fx = camera.GetK()(0,0);

  cv::Mat depth= cv::Mat::zeros(rgb.rows,rgb.cols, CV_32FC1);
  for(int r=0; r<rgb.rows; r++){
    for(int c=0; c<rgb.cols; c++){
      const float& disp = disparity.at<float>(r,c);
      if(disp < 1.)
        continue;
      depth.at<float>(r,c) = base_line *  fx / disp;
    }
  }
  _Put(gray, g_gray, depth, camera, rgb);
  return;
}

/*
void Seg::_Put_old(cv::Mat gray,
               cv::cuda::GpuMat g_gray,
               cv::Mat depth,
               const Camera& camera,
               cv::Mat rgb
               ) {

  auto start = std::chrono::steady_clock::now();

  std::vector<cv::Point2f> corners; {
    // ref: https://docs.opencv.org/3.4/d8/dd8/tutorial_good_features_to_track.html
    int maxCorners = 1000;
    double qualityLevel = 0.01;
    double minDistance = 10;
    int blockSize = 3, gradientSize = 3;
    bool useHarrisDetector = false;
    double k = 0.04;
    cv::goodFeaturesToTrack(gray, corners, maxCorners, qualityLevel, minDistance, cv::Mat(),
                            blockSize, gradientSize, useHarrisDetector,k);
  }
  if(corners.empty())
    return;
  if(gray0_.empty()){
    PutKeyframe(gray, g_gray);
    return;
  }

  cv::Mat flow = GetFlow(g_gray);
  const bool is_keyframe = IsKeyframe(flow,rgb);
  if(!is_keyframe){
    cv::waitKey(1);
    return;
  }
  g2o::SE3Quat Tc0c1 = TrackTc0c1(corners,flow,depth,camera);
  cv::Mat texture_edge = GetTextureEdge(gray);
  cv::Mat valid_mask = depth > 0.;

  cv::Mat exp_flow;
  GetExpectedFlow(camera, Tc0c1, depth, exp_flow, valid_mask);

  cv::Mat flow_scale; {
    std::vector<cv::Mat> tmp;
    cv::split(exp_flow,tmp);
    flow_scale = cv::abs(tmp[0]) + cv::abs(tmp[1]);
    // + max sampling???
  }

  cv::Mat flow_error2 = GetFlowError(flow, exp_flow, valid_mask);
  cv::Mat flow_error_scalar; {
    std::vector<cv::Mat> vec;
    cv::split(flow_error2, vec);
    flow_error_scalar = cv::abs( vec[0] ) + cv::abs( vec[1] );
  }
  cv::Mat flow_difference = GetDifference(flow);
  NormalizeScale(flow_scale, flow_difference, flow_error_scalar);
  if(is_keyframe)
    PutKeyframe(gray, g_gray);

  cv::Mat diff_edges = FlowDifference2Edge(flow_difference);
  cv::Mat expd_diffedges; {
    cv::Mat dist;
    cv::distanceTransform(~diff_edges, dist, cv::DIST_L2, cv::DIST_MASK_3, CV_32FC1);
    expd_diffedges = (dist < 20.) /255;
  }
  if(true){
    cv::Mat dst = GetColoredLabel(diff_edges);
    cv::addWeighted(rgb,.5,dst,1.,1.,dst);
    cv::imshow("diff_edges", dst);
  }
  cv::Mat error_edges = FlowError2Edge(flow_error_scalar, expd_diffedges, valid_mask);
  if(true){
    cv::Mat dst = GetColoredLabel(error_edges);
    cv::addWeighted(rgb,.5,dst,1.,1.,dst);
    cv::imshow("error_edges", dst);
  }
  cv::Mat marker = Segment(error_edges); // Sgement(error_edges,rgb);
#if 1
  std::map<int, ShapePtr> local_shapes = ConvertMarker2Instances(marker);
  static std::map<int, ShapePtr> global_shapes; // TODO 맴버변수로
  static int n_shapes = 0;
  const float min_iou = .3;
  std::map<int,int> l2g = TrackShapes(local_shapes, marker, flow, min_iou, global_shapes, n_shapes);
#endif
  if(false){
    cv::Mat dst = GetColoredLabel(marker);
    cv::addWeighted(rgb,.3, dst, .7, 1., dst);
    cv::Mat whites;
    cv::cvtColor(error_edges*255, whites, cv::COLOR_GRAY2BGR);
    cv::addWeighted(dst,1.,whites,.3, 1., dst);
    cv::imshow("Marker", dst);
  }

  if(false){
    cv::Mat n_sobel;
    cv::normalize(texture_edge, n_sobel, 0, 255, cv::NORM_MINMAX, CV_8UC1);

    cv::Mat div_norm = convertMat(flow_difference, 0., 1.);
    //cv::Mat div_norm; cv::normalize(divergence, div_norm, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    //cv::imshow("div_dst", div_norm);
    cv::Mat zeros = cv::Mat::zeros(texture_edge.rows, texture_edge.cols, CV_8UC1);
    std::vector<cv::Mat> vec = {n_sobel,div_norm, zeros};
    cv::Mat dst;
    cv::merge(vec, dst);
    cv::imshow("scores", dst);
  }
  if(false){
    cv::Mat dst;
    cv::Mat tmp = VisualizeFlow(exp_flow - flow);
    cv::addWeighted(rgb,.5,tmp,1.,1.,dst);
    cv::pyrDown(dst,dst);
    cv::imshow("flow_errors", dst);

    cv::Mat err_norm = convertMat(flow_error_scalar, 0., 1.);
    //cv::pyrDown(err_norm, err_norm);
    cv::imshow("nflow_errors", err_norm);
  }
  if(false){
    cv::Mat dst;
    cv::Mat tmp = VisualizeFlow(exp_flow);
    cv::addWeighted(rgb,.5,tmp,1.,1.,dst);
    cv::imshow("exp_flow", dst);
  }
  if(false){
    cv::Mat dst;
    cv::Mat tmp = VisualizeFlow(flow);
    cv::addWeighted(rgb,.5,tmp,1.,1.,dst);
    //cv::pyrDown(dst,dst);
    cv::imshow("flow", dst);
  }

  if(true){
    cv::Mat ndepth;
    cv::normalize(depth,ndepth,0,255,cv::NORM_MINMAX,CV_8UC1);
    cv::imshow("depth", ndepth);
  }
  return;
}

g2o::SE3Quat Seg::TrackTc0c1(const std::vector<cv::Point2f>& corners,
                             const cv::Mat flow,
                             const cv::Mat depth,
                             const Camera& camera) {
  const float fx = camera.GetK()(0,0);
  const float fy = camera.GetK()(1,1);
  const float cx = camera.GetK()(0,2);
  const float cy = camera.GetK()(1,2);

  std::vector< cv::Point2f > img0_points;
  std::vector< cv::Point3f > obj1_points;
  img0_points.reserve(corners.size());
  obj1_points.reserve(corners.size());
  for(const auto& pt1 : corners){
    const float& z1 = depth.at<float>(pt1);
    if(z1 < .001) // TODO invalid를 따로 분리, depth가 이상하다?
      continue;
    else if(z1 > 50.)
      continue;

    const auto& dpt01 = flow.at<cv::Point2f>(pt1); 
    const float duv = std::abs(dpt01.x)+std::abs(dpt01.y);
    //if(duv < 1. || duv > 50.)
    //  continue;
    float x1 = z1 * (pt1.x - cx) / fx;
    float y1 = z1 * (pt1.y - cy) / fy;
    img0_points.push_back(pt1-dpt01);
    obj1_points.push_back(cv::Point3f(x1,y1,z1));
  }

  const float max_rprj_err = 2.;
  cv::Mat cvK = (cv::Mat_<double>(3,3) << fx, 0., cx,
                                         0., fy, cy, 
                                         0., 0., 1. );
  cv::Mat cvDistortion = (cv::Mat_<double>(4,1) << 0., 0., 0., 0. );
  cv::Mat rvec = cv::Mat::zeros(3,1,cvK.type());
  cv::Mat tvec = cv::Mat::zeros(3,1,cvK.type());
  cv::Mat inliers;
  cv::solvePnPRansac(obj1_points,img0_points, cvK, cvDistortion, rvec, tvec, true, 100, max_rprj_err, 0.99, inliers);
  //cv::solvePnP(obj1_points, img0_points, cvK, cvDistortion, rvec, tvec, true);

  cv::Mat cvR;
  cv::Rodrigues(rvec, cvR);
  g2o::SE3Quat Tc0c1;
  Eigen::Matrix<double,3,3> R;
  R << cvR.at<double>(0,0), cvR.at<double>(0,1), cvR.at<double>(0,2),
    cvR.at<double>(1,0), cvR.at<double>(1,1), cvR.at<double>(1,2),
    cvR.at<double>(2,0), cvR.at<double>(2,1), cvR.at<double>(2,2);
  Eigen::Vector3d t(tvec.at<double>(0,0), tvec.at<double>(1,0), tvec.at<double>(2,0) );
  // std::cout << "t/frame = " << t.transpose() << std::endl; // Verbose translation per frame

  Eigen::Quaterniond quat(R);
  Tc0c1.setRotation(quat);
  Tc0c1.setTranslation(t);
  //std::cout << "Tc0c1 = " << Tc0c1.to_homogeneous_matrix() << std::endl;
  return Tc0c1;
}

*/

