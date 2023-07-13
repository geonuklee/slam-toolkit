#include <opencv2/core/types.hpp>
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
  //cv::Canny(gray, texture_edge, 300, 400);
  cv::Canny(gray, texture_edge, 100, 400);
  return texture_edge;
}

void Seg::NormalizeScale(const cv::Mat flow_scale,
                         cv::Mat& flow_difference, cv::Mat& flow_errors) {
  for(int r=0; r<flow_difference.rows; r++) {
    for(int c=0; c<flow_difference.cols; c++) {
      float& diff = flow_difference.at<float>(r,c);
      const float& s = flow_scale.at<float>(r,c);
      float S = std::min<float>(100.f,s);
      S = std::max<float>(1.f, S);
      float& err = flow_errors.at<float>(r,c);
      if(s > 1.) {
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

std::vector<cv::Point2f> SimplifyContour(const std::vector<cv::Point>& given_cnt){
  const float min_l = 20.;
  std::vector<cv::Point2f> simple_cnt;
  simple_cnt.reserve(given_cnt.size());

  const cv::Point2f pt0(given_cnt.begin()->x, given_cnt.begin()->y);
  simple_cnt.push_back(pt0);
  cv::Point2f pt_prev(pt0);
  for(const auto& _pt : given_cnt){
    cv::Point2f pt(_pt.x,_pt.y);
    cv::Point2f dpt = pt - pt_prev;
    //float l = cv::norm(dpt);
    float l = std::abs(dpt.x)+std::abs(dpt.y);
    if(l  < min_l)
      continue;
    simple_cnt.push_back(pt);
    pt_prev = pt;
  }

  if(simple_cnt.size() < 3)
    simple_cnt.clear();
  return simple_cnt;
}


std::map<int, ShapePtr> ConvertMarker2Instances(const cv::Mat marker) {
  cv::Mat fg = GetBoundary(marker) < 1;
  const int mode   = cv::RETR_TREE;
  const int method = cv::CHAIN_APPROX_SIMPLE;
  std::vector<std::vector<cv::Point> > contours;
  std::vector<cv::Vec4i> hierarchy;
  cv::findContours(fg,contours,hierarchy,mode,method);

  std::map<int, ShapePtr > shapes;
  std::map<int, float> areas;
  for(int i = 0; i < contours.size(); i++){
    // Ref) https://076923.github.io/posts/Python-opencv-21/
    // const cv::Vec4i& h = hierarchy.at(i); // h[0:3] : '이전윤곽', '다음윤곽', '내곽윤곽', '외곽윤곽(부모)'
    const std::vector<cv::Point>& _contour = contours.at(i);
    std::vector<cv::Point2f> contour = SimplifyContour(_contour);
    if(contour.empty())
      continue;
    const int& l = marker.at<int>(*contour.begin()); // Binary이미지에서 contour를 따내므로, 경계선 문제없이 라빌 획득됨.
    if(l < 1)
      continue;
    const float area = cv::contourArea(contour);
    if(area < 50)
      continue;
    if( !shapes.count(l) ){
      shapes[l] = std::make_shared<Shape>();
      areas[l]  = area;
    }
    else if(areas.at(l) > area)
      continue;
    ShapePtr s_ptr = shapes[l];
    s_ptr->label_ = l;
    s_ptr->outerior_ = contour;
  }
  for(auto it : shapes)
    it.second->UpdateBB();
  return shapes;
}

void Shape::UpdateBB() {
  {
    cv::Point2f x0(9999999.f,9999999.f);
    cv::Point2f x1(-x0);
    for(const auto& pt : outerior_){
      x0.x = std::min<float>(x0.x, pt.x);
      x0.y = std::min<float>(x0.y, pt.y);
      x1.x = std::max<float>(x1.x, pt.x);
      x1.y = std::max<float>(x1.y, pt.y);
    }
    outerior_bb_ = cv::Rect2f(x0.x,x0.y,x1.x-x0.x,x1.y-x0.y);
  }
  return;
}

static bool BbCollision(const cv::Rect2f& bb, const float& x, const float& y) {
  //bb.width;
  if(x < bb.x )
    return false;
  if(x > bb.x+bb.width)
    return false;
  if(y < bb.y)
    return false;
  if(y > bb.y+bb.width)
    return false;
  return true;
}

bool Shape::HasCollision(const int& _x, const int& _y, bool check_contour) const {
   
  // *[x] global shape에 대해 BB 충돌체크.
  // *[x] contour에 대해 충돌체크
  const float x = _x;
  const float y = _y;
  cv::Point2f pt(x,y);

  if( !BbCollision(outerior_bb_, x, y) )
    return false;
  if(!check_contour)
    return true;

  // Interior contour과 충돌 없는 경우, outerior contour와 충돌체크
  bool b_outerior_collision = cv::pointPolygonTest(outerior_, pt, false) > 0.;
  return b_outerior_collision;
}

cv::Mat VisualizeTrackedShapes(const std::map<int, ShapePtr>& global_shapes,
                               const cv::Mat local_marker){
  cv::Mat dst = cv::Mat::zeros(local_marker.rows, local_marker.cols, CV_8UC3);
  for(auto it : global_shapes){
    ShapePtr ptr = it.second;
    if(!ptr->stabilized_)
      continue;
    std::vector< std::vector<cv::Point> > cnts;
    cnts.resize(1);
    cnts[0].reserve(ptr->outerior_.size() );
    for( auto pt: ptr->outerior_)
      cnts[0].push_back(cv::Point(pt.x,pt.y));
    const auto& color = colors.at(it.first % colors.size() );
    cv::drawContours(dst, cnts, 0, color, 2);
  }

  return dst;
}

std::map<int,int> TrackShapes(const std::map<int, ShapePtr>& local_shapes,
                              const cv::Mat& local_marker,
                              const cv::Mat& flow,
                              std::map<int, ShapePtr>& global_shapes,
                              int& n_shapes) {
  std::map<int,int> matches;
  // TODO 함수 가장 마지막과 병합
  if(global_shapes.empty()){
    for(auto it : local_shapes){
      const int gid =  ++n_shapes;
      it.second->label_ = gid;
      global_shapes[gid] = it.second;
      //matches[gid] = gid;
    }
    return matches;
  }

  /* [x] global_shape의 motion update
   * [x] IoU 계산후 best matching 획득.
   * [x] Missing shape에 대한 대처 : 너무큰건 거르고, 자주 발견되는걸 믿자.
  */
  cv::Mat flow0 = cv::Mat::zeros(flow.rows,flow.cols,CV_32FC2);
  for(int r1=0; r1<flow.rows; r1++){
    for(int c1=0; c1<flow.cols; c1++){
      // dpt01 : {1} coordinate에서 0->1 변위 벡터, 
      const auto& dpt01 = flow.at<cv::Point2f>(r1,c1);
      if(std::abs(dpt01.x)+std::abs(dpt01.y) < 1e-10)
        continue;
      cv::Point2f pt0(c1-dpt01.x, r1-dpt01.y);
      if(pt0.x < 0 || pt0.y < 0 || pt0.x > flow.cols-1 || pt0.y > flow.rows-1)
        continue;
      flow0.at<cv::Point2f>(pt0) = dpt01;
    }
  }

  // Motion update
  for(auto git : global_shapes){
    ShapePtr ptr = git.second;
    for(auto& pt : ptr->outerior_){
      const auto& dpt01 = flow0.at<cv::Point2f>(pt);
      pt += dpt01;
      // TODO 화면 구석에 몰리면 지워야하나?
      pt.x = std::max<float>(pt.x, 0.f);
      pt.x = std::min<float>(pt.x,local_marker.cols-1.f);
      pt.y = std::max<float>(pt.y, 0.f);
      pt.y = std::min<float>(pt.y,local_marker.rows-1.f);
    }
    ptr->UpdateBB();
  }

  // Sampling area
  std::map<int, size_t> g_areas;
  std::map<int, size_t> l_areas;
  std::map<int, std::map<int, size_t>  > l2g_overlaps;
  bool check_contour = true;
  const int dl = 2;
  for(int r=dl; r+dl<local_marker.rows; r+=dl){
    for(int c=dl; c+dl<local_marker.cols; c+=dl){
      const int& local_l = local_marker.at<int>(r,c);
      if(local_l < 1)
        continue;
      l_areas[local_l]++;
      for(auto git : global_shapes){
        ShapePtr ptr = git.second;
        bool b = ptr->HasCollision(c,r, check_contour);
        if(!b)
          continue;
        // TODO 겹치는 경우 
        const int& global_l = ptr->label_;
        g_areas[global_l]++;
        l2g_overlaps[local_l][global_l]++;
      }
    }
  }

  // 각 local_l 에 대해 가장 높은 IoU를 보이는 global_l을 연결
  std::map<int, std::pair<int, float > > g2l;
  for(auto it : l2g_overlaps){
    const int& local_l = it.first;
    float n_local = l_areas.at(local_l);
    //std::cout << "l#" << local_l << " : {";
    float max_iou = .5; // min_iou
    int best_gid = -1;
    for(auto it2 : it.second){
      const int& global_l = it2.first;
      float n_overlap = it2.second;
      float n_global = g_areas.at(global_l);
      float iou = n_overlap / (n_global+n_local-n_overlap);
      //std::cout << "(" << global_l << "," << iou << "),";
      if(iou < max_iou)
        continue;
      max_iou = iou;
      best_gid = global_l;
    }
    //std::cout << "}" << std::endl;
    if(best_gid < 0)
      continue;
    if(g2l.count(best_gid)){
      const auto& ll_iou = g2l.at(best_gid);
      if(max_iou > ll_iou.second)
        g2l[best_gid] = std::make_pair(local_l, max_iou);
    } else
      g2l[best_gid] = std::make_pair(local_l, max_iou);
  }
  // g2l에 g->l l->g 연결에서 모두 IoU 가 최대인 케이스만 남음.
  std::list<int> layoff_list;
  for(auto it : global_shapes){
    const int& gid = it.first;
    ShapePtr g_ptr = it.second;
    if(g2l.count(gid)){
      const int& lid = g2l.at(gid).first;
      ShapePtr l_ptr = local_shapes.at(lid);
      g_ptr->n_missing_ = 0;
      g_ptr->n_matching_++;
      g_ptr->n_belief_++;
      if(g_ptr->n_belief_ > 1)
        g_ptr->stabilized_ = true;
      g_ptr->outerior_ = l_ptr->outerior_;
      g_ptr->outerior_bb_ = l_ptr->outerior_bb_;
      matches[lid] = gid;
      continue;
    }
    g_ptr->n_missing_++;
    g_ptr->n_belief_--;

    if(g_ptr->n_missing_ > 2)
      layoff_list.push_back(gid);
    else if(g_ptr->n_belief_ < 1)
      layoff_list.push_back(gid);

  }
  for(int gid : layoff_list)
    global_shapes.erase(gid);
  // 매칭안된 local shape를 새로 등록
  for(auto it : local_shapes){
    const int& lid = it.first;
    ShapePtr l_ptr = it.second;
    if(matches.count(lid))
      continue;
    const int gid =  ++n_shapes;
    l_ptr->label_ = gid;
    global_shapes[gid] = l_ptr;
    //matches[gid] = gid;
  }
  return matches;
}

void Seg::_Put(cv::Mat gray,
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
  //cv::Mat valid_mask = cv::Mat::ones(gray.rows, gray.cols, CV_8UC1);
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

    //cv::Mat dst = GetColoredLabel(diff_edges);
    //cv::imshow("diff_edges", dst);
  }
  cv::Mat error_edges = FlowError2Edge(flow_error_scalar, expd_diffedges, valid_mask); {
    //cv::Mat dst = GetColoredLabel(error_edges);
    //cv::addWeighted(rgb,.5,dst,1.,1.,dst);
    //cv::imshow("error_edges", dst);
  }
  cv::Mat marker = Segment(error_edges); // Sgement(error_edges,rgb);
#if 1
  std::map<int, ShapePtr> local_shapes = ConvertMarker2Instances(marker);
  static std::map<int, ShapePtr> global_shapes; // TODO 맴버변수로
  static int n_shapes = 0;
  std::map<int,int> l2g = TrackShapes(local_shapes, marker, flow, global_shapes, n_shapes);
  {
    // TODO 결과물의 visualization
    std::cout << "n(gShapes) = " << global_shapes.size() << std::endl;
    cv::Mat dst = VisualizeTrackedShapes(global_shapes, marker);
    cv::imshow("TrackedShapes", dst);
  }
#endif

  {
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
    cv::imshow("rgb", rgb);
    cv::imshow("depth", ndepth);
  }
  return;
}

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

  /*
  cv::imshow("rgb", rgb);
  cv::imshow("rgb_r", rgb_r);
  cv::Mat ndepth;
  cv::normalize(depth,ndepth,0,255,cv::NORM_MINMAX,CV_8UC1);
  cv::imshow("depth", ndepth);
  */
  _Put(gray, g_gray, depth, camera, rgb);
  return;
}


void Seg::Put(cv::Mat rgb, cv::Mat depth, const DepthCamera& camera) {
  cv::Mat gray;
  cv::cvtColor(rgb,gray,cv::COLOR_BGR2GRAY);
  cv::cuda::GpuMat g_gray;
  g_gray.upload(gray);

  _Put(gray, g_gray, depth, camera, rgb);
  return;
}
