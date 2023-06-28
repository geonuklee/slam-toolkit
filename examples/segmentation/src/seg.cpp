#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudastereo.hpp>
#include <opencv2/cudafilters.hpp>

#include "../include/seg.h"
#include "../include/util.h"

Seg::Seg() {
  //optical_flow_ = cv::cuda::FarnebackOpticalFlow::create(5, 0.5, true, 15); // 30
  optical_flow_ = cv::cuda::BroxOpticalFlow::create(0.197f, 50.0f, 0.8f, 5, 150, 10); // 가장좋은결과.
  //opticalflow_   = cv::cuda::OpticalFlowDual_TVL1::create();
  //opticalflow_ = cv::cuda::DensePyrLKOpticalFlow::create(cv::Size(13,13),3);
}


bool Seg::IsKeyframe(cv::Mat flow, cv::Mat rgb) {
#if 1
  return true;
#else
  const float flow_threshold = 20.;
  bool is_keyframe = false;
  // Keyframe 판정을 위한 'Median flow' 획득.
  std::vector<float> norm_flows;
  norm_flows.reserve(corners0_.size());
  for(const cv::Point2f& pt0 : corners0_){
    const cv::Point2f& dpt = flow.at<cv::Point2f>(pt0);
    if(dpt.x==0. && dpt.y==0.)
      continue;
    norm_flows.push_back(cv::norm(dpt));
  }
  std::sort(norm_flows.begin(),norm_flows.end(), std::greater<int>());
  int n = .2 * norm_flows.size();
  float median = norm_flows.at(n);
  is_keyframe = median > flow_threshold;
  if(!rgb.empty()){
    cv::Mat dst = rgb.clone();
    const auto& color = is_keyframe?CV_RGB(255,0,0):CV_RGB(0,0,255);
    for(const cv::Point2f& pt0 : corners0_){
      const cv::Point2f& dpt = flow.at<cv::Point2f>(pt0);
      cv::arrowedLine(dst, pt0, pt0+dpt, color, 1); // verbose
    }
    cv::imshow("FeatureTrack", dst);//verbose
  }
  return is_keyframe;
#endif
}

void Seg::PutKeyframe(cv::Mat gray, cv::cuda::GpuMat g_gray){
  gray0_ = gray;
  g_gray0_ = g_gray;
  return;
}

cv::Mat Seg::GetFlow(cv::cuda::GpuMat g_gray) {
  cv::Mat flow;
  cv::cuda::GpuMat g_flow;
  // ref) https://android.googlesource.com/platform/external/opencv3/+/master/samples/gpu/optical_flow.cpp#189
  if(optical_flow_->getDefaultName() == "DenseOpticalFlow.BroxOpticalFlow"){
    cv::cuda::GpuMat g_f_gray,g_f_gray0;
    g_gray0_.convertTo(g_f_gray0,CV_32FC1, 1./255.);
    g_gray.convertTo(g_f_gray,CV_32FC1, 1./255.);
    optical_flow_->calc(g_f_gray, g_f_gray0, g_flow);
  }
  else{
    optical_flow_->calc(g_gray, g_gray0_, g_flow);
  }
  g_flow.download(flow); // flow : 2ch float image
  flow = -flow;
  return flow;
}

cv::Mat Seg::GetTextureEdge(cv::Mat gray) {
  cv::Mat texture_edge;
#if 1
    //cv::Canny(gray, texture_edge, 300, 400);
    cv::Canny(gray, texture_edge, 100, 400);
#else
    cv::cuda::GpuMat g1,g2,g3, g_all;
    static auto filter1 = cv::cuda::createSobelFilter(CV_8UC1,CV_32FC1,1,1);
    static auto filter2 = cv::cuda::createSobelFilter(CV_8UC1,CV_32FC1,1,0);
    static auto filter3 = cv::cuda::createSobelFilter(CV_8UC1,CV_32FC1,0,1);
    filter1->apply(g_gray, g1);
    filter2->apply(g_gray, g2);
    filter3->apply(g_gray, g3);
    cv::Mat s1,s2,s3;
    g1.download(s1);
    g2.download(s2);
    g3.download(s3);
    /*
    sobel = cv::abs(s1)+cv::abs(s2)+cv::abs(s3);
    double minValue, maxValue;
    cv::Point minLoc, maxLoc;
    cv::minMaxLoc(sobel, &minValue, &maxValue, &minLoc, &maxLoc);
    sobel = sobel > .1*maxValue;
    */
    float threshold = 100.;
    cv::bitwise_or(cv::abs(s1)>threshold, cv::abs(s2)>threshold, texture_edge);
    cv::bitwise_or(texture_edge, cv::abs(s3)>threshold, texture_edge);
#endif
    return texture_edge;
}

void Seg::NormalizeScale(const cv::Mat disparity, const cv::Mat flow_scale,
                         cv::Mat& flow_difference, cv::Mat& flow_errors) {
  for(int r=0; r<flow_difference.rows; r++) {
    for(int c=0; c<flow_difference.cols; c++) {
      float& diff = flow_difference.at<float>(r,c);
      const float& s = flow_scale.at<float>(r,c);
      float S = std::min<float>(100.f,s);
      S = std::max<float>(1.f, S);
      //const float S = 1.;
      const float& disp = disparity.at<float>(r,c);
      float& err = flow_errors.at<float>(r,c);
      if(s > 1.) { //  && disp > 1.){
        diff /= S;
        err /= S;
      }
      else{
        diff = 0.;
        err = 0.;
      }
    }
  }
  return;
}


g2o::SE3Quat Seg::TrackTc0c1(const std::vector<cv::Point2f>& corners,
                             const cv::Mat flow,
                             const cv::Mat disparity,
                             const StereoCamera& camera) {
  // corners : {1} coordinate의 특징점
  // flow : {1} coordinate에서 0->1 변위 벡터, 
  // disparity : {1} coordinate의 disparity
  const auto Trl_ = camera.GetTrl();
  const float base_line = -Trl_.translation().x();
  const float fx = camera.GetK()(0,0);
  const float fy = camera.GetK()(1,1);
  const float cx = camera.GetK()(0,2);
  const float cy = camera.GetK()(1,2);
  // std::cout << "baseline = " << base_line << std::endl;
  std::vector< cv::Point2f > img0_points;
  std::vector< cv::Point3f > obj1_points;
  img0_points.reserve(corners.size());
  obj1_points.reserve(corners.size());
  for(const auto& pt1 : corners){
    const float& disp = disparity.at<float>(pt1);
    if(disp < 1.)
      continue;
    const auto& dpt01 = flow.at<cv::Point2f>(pt1); 
    const float duv = std::abs(dpt01.x)+std::abs(dpt01.y);
    //if(duv < 1. || duv > 50.)
    //  continue;
    float z1 = base_line *  fx / disp;
    float x1 = z1 * (pt1.x - cx) / fx;
    float y1 = z1 * (pt1.y - cy) / fy;
    img0_points.push_back(pt1-dpt01);
    //img0_points.push_back(pt1);
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
  //std::cout << "tvec = " << tvec.t() << std::endl;
  //std::cout << "t = " << t.transpose() << std::endl;
  Eigen::Quaterniond quat(R);
  Tc0c1.setRotation(quat);
  Tc0c1.setTranslation(t);
  //std::cout << "Tc0c1 = " << Tc0c1.to_homogeneous_matrix() << std::endl;
  return Tc0c1;
}

void Seg::Put(cv::Mat gray, cv::Mat gray_r, const StereoCamera& camera) {
  if(camera.GetD().norm() > 1e-5){
    std::cerr << "Not support distorted image. Put rectified image" << std::endl;
    exit(1);
  }
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

  cv::Mat rgb;
  cv::cvtColor(gray, rgb, cv::COLOR_GRAY2BGR);

  cv::cuda::GpuMat g_gray;
  g_gray.upload(gray);
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

  cv::cuda::GpuMat g_gray_r;
  g_gray_r.upload(gray_r);
  cv::Mat disparity = GetDisparity(g_gray,g_gray_r);

#if 1
  g2o::SE3Quat Tc0c1 = TrackTc0c1(corners,flow,disparity,camera); // TODO 이걸로 대신한다.
#else
  g2o::SE3Quat Tc0c1 = Tc0w_ * Tcw.inverse(); // 주어진 Tcw를 사용.
#endif
  cv::Mat texture_edge = GetTextureEdge(gray);
  cv::Mat texture_mask; {
    /* 
      1) small_r보다 작은 edge를 지우기 위해,
         dist < 5.
    */
    cv::Mat dist1;
    cv::distanceTransform(~texture_edge, dist1, cv::DIST_L2, cv::DIST_MASK_3, CV_32FC1);
    cv::Mat labels, stats, centroids;
    cv::connectedComponentsWithStats(dist1 < 2.,labels,stats,centroids);
    std::set<int> inliers;
    for(int i = 1; i < stats.rows; i++){
      const int max_wh = std::max(stats.at<int>(i,cv::CC_STAT_WIDTH),
                                  stats.at<int>(i,cv::CC_STAT_HEIGHT));
      if(max_wh > 50)
        inliers.insert(i);
    }

    for(int r = 0; r < labels.rows; r++){
      for(int c = 0; c < labels.cols; c++){
        int& l = labels.at<int>(r,c);
        l = inliers.count(l) ? 1 : 0;
      }
    }
    cv::Mat dist2;
    cv::distanceTransform(labels<1, dist2, cv::DIST_L2, cv::DIST_MASK_3, CV_32FC1);
    texture_mask = dist2 < 20.; // r1+r2
    //cv::imshow("texture_mask", 255*texture_mask);
  }
  //cv::Mat valid_mask = texture_mask.clone();
  cv::Mat valid_mask = cv::Mat::ones(texture_mask.rows, texture_mask.cols, CV_8UC1);
  //cv::bitwise_and(valid_mask, disparity>2.,valid_mask);
  //cv::bitwise_and(valid_mask, disparity<40.,valid_mask);

  cv::Mat exp_flow;
  GetExpectedFlow(camera, Tc0c1, disparity, exp_flow, valid_mask);

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
  cv::Mat flow_difference = GetDifference(flow, disparity);

  NormalizeScale(disparity, flow_scale, flow_difference, flow_error_scalar);
  if(is_keyframe)
    PutKeyframe(gray, g_gray);
  //static int i = 0;
  //if( ++i < 5)
  //  return;

  cv::Mat diff_edges = FlowDifference2Edge(flow_difference);
  cv::Mat expd_diffedges; {
    cv::Mat dist;
    cv::distanceTransform(~diff_edges, dist, cv::DIST_L2, cv::DIST_MASK_3, CV_32FC1);
    expd_diffedges = (dist < 20.) /255;
    cv::Mat dst = GetColoredLabel(diff_edges);
    cv::imshow("diff_edges", dst);
  }
  cv::Mat error_edges = FlowError2Edge(flow_error_scalar, expd_diffedges, valid_mask);
  //cv::bitwise_or(error_edges, diff_edges, error_edges); // 어두워서 disp가 안잡히는 부분에서만 도움이된다.
  {
    cv::Mat dst = GetColoredLabel(error_edges);
    cv::addWeighted(rgb,.5,dst,1.,1.,dst);
    cv::imshow("edges", dst);
  }

  cv::Mat marker = Segment(error_edges); // Sgement(error_edges,rgb);
  {
    cv::Mat dst = GetColoredLabel(marker);
    cv::addWeighted(rgb,.3, dst, .7, 1., dst);
    cv::Mat whites;
    cv::cvtColor(error_edges*255, whites, cv::COLOR_GRAY2BGR);
    cv::addWeighted(dst,1.,whites,.3, 1., dst);
    cv::imshow("Marker", dst);
  }

  if(false){
    cv::Mat flow_dist;
    cv::distanceTransform(flow_difference < .5, flow_dist, cv::DIST_L2, cv::DIST_MASK_3);
    cv::Mat output_edges = cv::Mat::zeros(texture_edge.rows, texture_edge.cols, CV_8UC1);
    for(int r=0; r<output_edges.rows; r++){
      for(int c=0; c<output_edges.cols;c++){
        const uchar& te = texture_edge.at<uchar>(r,c);
        if(!te)
          continue;
        float R = 1.*disparity.at<float>(r,c);
        if(flow_dist.at<float>(r,c) > R)
          continue;
        output_edges.at<uchar>(r,c) = 1;
      }
    }
    cv::imshow("output", 255*output_edges);
    //cv::imshow("flow_dist", 255*(flow_dist<5.));
    //cv::Mat texture_dist;
    //cv::distanceTransform(texture_edge < 1, texture_dist, cv::DIST_L2, cv::DIST_MASK_3);
    //cv::imshow("texture_dist", 255*(texture_dist<5.) );
    cv::Mat zeros = cv::Mat::zeros(texture_edge.rows, texture_edge.cols, CV_8UC1);
    std::vector<cv::Mat> vec = {zeros,flow_dist<2.,texture_edge>0};
    cv::Mat dst;
    cv::merge(vec, dst);
    dst *= 255;
    cv::imshow("thick_edges", dst);
  }

  if(true){
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
  if(true){
    cv::Mat dst;
    cv::Mat tmp = VisualizeFlow(exp_flow - flow);
    cv::addWeighted(rgb,.5,tmp,1.,1.,dst);
    cv::pyrDown(dst,dst);
    cv::imshow("flow_errors", dst);

    cv::Mat err_norm = convertMat(flow_error_scalar, 0., 1.);
    //cv::pyrDown(err_norm, err_norm);
    cv::imshow("nflow_errors", err_norm);
  }
  if(true){
    cv::Mat dst;
    cv::Mat tmp = VisualizeFlow(exp_flow);
    cv::addWeighted(rgb,.5,tmp,1.,1.,dst);
    cv::imshow("exp_flow", dst);
  }
  if(true){
    cv::Mat dst;
    cv::Mat tmp = VisualizeFlow(flow);
    cv::addWeighted(rgb,.5,tmp,1.,1.,dst);
    //cv::pyrDown(dst,dst);
    cv::imshow("flow", dst);
  }
  if(true){
    cv::Mat dst;
    disparity.convertTo(dst, CV_8UC1);
    cv::imshow("disp",  dst);
    /*
    cv::Mat ndisp;
    cv::normalize(disparity, ndisp, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::pyrDown(ndisp,ndisp);
    cv::imshow("disp", ndisp);
    */
  }
  auto end = std::chrono::steady_clock::now();
  auto etime_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  std::cout << "etime = " << etime_ms << std::endl;
  return;
}
