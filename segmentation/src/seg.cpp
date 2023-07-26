#include <opencv2/core/types.hpp>
#include <opencv2/cudaoptflow.hpp>
// #include <opencv2/cudafilters.hpp>

#include "../include/seg.h"
#include "../include/util.h"
#include "orb_extractor.h"

Seg::Seg() {
  //optical_flow_ = cv::cuda::FarnebackOpticalFlow::create(5, 0.5, true, 15); // 30
  optical_flow_ = cv::cuda::BroxOpticalFlow::create(0.197f, 50.0f, 0.8f, 5, 150, 10); // 가장좋은결과.
  //optical_flow_   = cv::cuda::OpticalFlowDual_TVL1::create();
  //optical_flow_ = cv::cuda::DensePyrLKOpticalFlow::create(cv::Size(13,13),3);
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

std::map<int,int> TrackShapes(const std::map<int, ShapePtr>& local_shapes,
                              const cv::Mat& local_marker,
                              const cv::Mat& flow,
                              const float min_iou,
                              std::map<int, ShapePtr>& global_shapes,
                              int& n_shapes) {
  // Convert coordinate of flow
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
      if(local_l > 0)
        l_areas[local_l]++;
      for(auto git : global_shapes){
        ShapePtr g_ptr = git.second;
        bool b = g_ptr->HasCollision(c,r, check_contour);
        if(!b)
          continue;
        const int& global_l = g_ptr->label_;
        g_areas[global_l]++;
        if(local_l > 0)
          l2g_overlaps[local_l][global_l]++;
      }
    }
  }

  // 각 local_l 에 대해 가장 높은 IoU를 보이는 global_l을 연결
  std::map<int, std::pair<int, float > > g2l_scores;
  for(auto it : l2g_overlaps){
    const int& local_l = it.first;
    float n_local = l_areas.at(local_l);
    //std::cout << "l#" << local_l << " : {";
    float max_iou = min_iou; // min_iou
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
    if(g2l_scores.count(best_gid)){
      const auto& prev_candidate = g2l_scores.at(best_gid);
      if(max_iou > prev_candidate.second)
        g2l_scores[best_gid] = std::make_pair(local_l, max_iou);
    } else if(best_gid > 0)
      g2l_scores[best_gid] = std::make_pair(local_l, max_iou);
  }

  // g2l에 g->l l->g 연결에서 모두 IoU 가 최대인 케이스만 남기기.
  std::list<int> layoff_list;
  std::map<int,int> matches;
  for(auto it : global_shapes){
    const int& gid = it.first;
    ShapePtr g_ptr = it.second;
    if(g2l_scores.count(gid) ){ 
      const int& lid = g2l_scores.at(gid).first;
      if(local_shapes.count(lid)){
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
    }

    /*
    case 1) Not g2l_socres.count(gid) 
      -> min_iou를 넘기는 correspondence가 없는 global shape
    case 2) Not local_shapes.count(lid)
      -> Optimal correspondence(lid) 가
         ConvertMarker2Instances에서 필터링된 너무 작은 instance
    */
    g_ptr->n_missing_++;
    g_ptr->n_belief_--;
    if(g_ptr->n_missing_ > 2 ||  g_ptr->n_belief_ < 1)
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
    //matches[lid] = gid // 이걸 취급해야할진 모르겠다.
  }
  return matches;
}


cv::Mat GetGrad(const cv::Mat depth , const cv::Mat valid_mask,
             const Camera& camera,
             cv::Mat& gradx,
             cv::Mat& grady,
             cv::Mat& valid_grad,
             bool verbose
             ) {
  gradx = cv::Mat::zeros(depth.rows, depth.cols, CV_32FC1);
  grady = cv::Mat::zeros(depth.rows, depth.cols, CV_32FC1);
  valid_grad = cv::Mat::zeros(depth.rows, depth.cols, CV_8UC1);
  // sample offset, [meter]
  const float s = 0.1;
  const float sfx = s*camera.GetK()(0,0);
  const float sfy = s*camera.GetK()(1,1);
  cv::Mat dst;
  if(verbose)
    dst = cv::Mat::zeros(depth.rows, depth.cols, CV_8UC3);
  for(int r = 0; r < depth.rows; r++) {
    for(int c = 0; c < depth.cols; c++) {
      const cv::Point2i pt(c,r);
      if(!valid_mask.at<uchar>(pt))
        continue;
      const float& z = depth.at<float>(pt);
      const int du = std::max<int>( (int) sfx/z, 1);
      const int dv = std::max<int>( (int) sfy/z, 1);
      cv::Point2i ptx0(c-du,r);
      cv::Point2i ptx1(c+du,r);
      cv::Point2i pty0(c,r-dv);
      cv::Point2i pty1(c,r+dv);
      if(ptx0.x < 0 || !valid_mask.at<uchar>(ptx0) )
        continue;
      if(ptx1.x > depth.cols || !valid_mask.at<uchar>(ptx1) )
        continue;
      if(pty0.y < 0 || !valid_mask.at<uchar>(pty0) )
        continue;
      if(pty1.y > depth.rows || !valid_mask.at<uchar>(pty1) )
        continue;
      valid_grad.at<uchar>(pt) = true;
      const float gx = (depth.at<float>(ptx1) - depth.at<float>(ptx0) ) / s;
      const float gy = (depth.at<float>(pty1) - depth.at<float>(pty0) ) / s;
      gradx.at<float>(pt) = gx;
      grady.at<float>(pt) = gy;
      if(!verbose)
        continue;
      auto& color = dst.at<cv::Vec3b>(pt);
      int val = std::min<int>(255, std::max<int>(0, std::abs(10.*gy) ) );
      if(gy > 0)
        color[0] = val;
      else
        color[1] = val;
      val = std::min<int>(255, std::max<int>(0, std::abs(10.*gx) ) );
      color[2] = val;
    }
  }
  return dst;
}

static bool SameSign(const float& v1, const float& v2){
  if(v1 > 0.)
    return v2 > 0.;
  else if(v1 < 0.)
    return v2 < 0.;
  return (v1 == 0.) && (v2 == 0.);
}

cv::Mat GetConcaveEdges(const cv::Mat gradx,
                     const cv::Mat grady,
                     const cv::Mat depth,
                     const cv::Mat valid_mask,
                     const Camera& camera){
  cv::Mat hessian = cv::Mat::zeros(depth.rows, depth.cols, CV_32FC1);
  const int dl = 5; // "Shallow groove"를 찾을것도 아닌데, 간격이 좁을필요가..
  const int R = 10; // NMAS range
  /*
  const float s = 1. / ( 2. * (float)dl );
  const float sfx = s*camera.GetK()(0,0);
  const float sfy = s*camera.GetK()(1,1);
  */
  const float fx = camera.GetK()(0,0);
  const float fy = camera.GetK()(1,1);

  for(int r=dl; r < depth.rows-dl; r++) {
    for(int c=dl; c < depth.cols-dl; c++) {
      const cv::Point2i pt(c,r);
      if(!valid_mask.at<uchar>(pt))
        continue;
      const float& z = depth.at<float>(pt);
      cv::Point2i ptx0(c-dl,r);
      cv::Point2i ptx1(c+dl,r);
      cv::Point2i pty0(c,r-dl);
      cv::Point2i pty1(c,r+dl);
      if(ptx0.x < 0 || !valid_mask.at<uchar>(ptx0) )
        continue;
      if(ptx1.x > depth.cols || !valid_mask.at<uchar>(ptx1) )
        continue;
      if(pty0.y < 0 || !valid_mask.at<uchar>(pty0) )
        continue;
      if(pty1.y > depth.rows || !valid_mask.at<uchar>(pty1) )
        continue;
      float hxx = gradx.at<float>(ptx1) - gradx.at<float>(ptx0);
      float hyy = grady.at<float>(pty1) - grady.at<float>(pty0);
      hessian.at<float>(pt) = hxx + hyy;
    }
  }
  std::vector<cv::Point2i> samples = {
    cv::Point2i(1,0),
    cv::Point2i(-1,0),
    cv::Point2i(0,1),
    cv::Point2i(0,-1),
  };

  cv::Mat edges = cv::Mat::zeros(depth.rows, depth.cols, CV_8UC1);
  for(int r=dl; r < depth.rows-dl; r++) {
    for(int c=dl; c < depth.cols-dl; c++) {
      const cv::Point2i pt(c,r);
      if(!valid_mask.at<uchar>(pt))
        continue;
      const float& h = hessian.at<float>(pt);
      bool ismax = true;
      for(int dr=1; dr<R; dr++){
        for(const auto& dpt : samples){
          const cv::Point2i pt1 = pt + dr * dpt;
          const float& h1 = hessian.at<float>(pt1);
          if(std::abs(h1) < std::abs(h))
            continue;
          if(SameSign(h1,h))
            continue;
          ismax = false;
          break;
        }
        if(!ismax)
          break;
      }
      if(!ismax) // NMAS
        continue;
      // Hessian을 고려한 edge 판단
      if( h < -15.)
        edges.at<uchar>(pt) = 1;
    }
  }
  return edges;
}

cv::Mat FilterThinNoise(const cv::Mat edges){
  cv::Mat kernel = cv::Mat::ones(3,3,edges.type());
  cv::Mat output;
  cv::morphologyEx(edges, output, cv::MORPH_OPEN, kernel );
  return output;
}

cv::Mat GetDDEdges(const cv::Mat depth, const cv::Mat valid_mask,
                   const Camera& camera){
  cv::Mat edge = cv::Mat::zeros(depth.rows, depth.cols, CV_8UC1);
  std::vector<cv::Point2i> samples = {
    cv::Point2i(1,0),
    cv::Point2i(-1,0),
    cv::Point2i(0,1),
    cv::Point2i(0,-1),
  };

  for(int r0 = 0; r0 < depth.rows; r0++) {
    for(int c0 = 0; c0 < depth.cols; c0++) {
      const cv::Point2i pt0(c0,r0);
      if(!valid_mask.at<unsigned char>(pt0))
        continue;
      const float& z0 = depth.at<float>(pt0);
      unsigned char& e = edge.at<unsigned char>(pt0);
      for(int l=1; l < 4; l++){
        for(const auto& dpt : samples){
          const cv::Point2i pt1 = pt0+l*dpt;
          if(pt1.x < 0 || pt1.x >= depth.cols || pt1.y < 0 || pt1.y >= depth.rows)
            continue;
          if(!valid_mask.at<unsigned char>(pt1))
            continue;
          const float& z1 = depth.at<float>(pt1);
          float th = std::max<float>(0.01*z0, 2.);
          if(std::abs(z1-z0) < th)
            continue;
          e = true;
          break;
        }
        if(e)
          break;
      }
    }
  }
  return edge;
}

std::vector<cv::Point2f> GetCorners(const cv::Mat gray) {
  // ref: https://docs.opencv.org/3.4/d8/dd8/tutorial_good_features_to_track.html
  std::vector<cv::Point2f> corners;
  int maxCorners = 1000;
  double qualityLevel = 0.01;
  double minDistance = 10;
  int blockSize = 3, gradientSize = 3;
  bool useHarrisDetector = false;
  double k = 0.04;
  cv::goodFeaturesToTrack(gray, corners, maxCorners, qualityLevel, minDistance, cv::Mat(),
                          blockSize, gradientSize, useHarrisDetector,k);
  return corners;
}

void HighlightValidmask(const cv::Mat valid_mask, cv::Mat& rgb){
  const int diff = 100;
  for(int r=0; r<valid_mask.rows;r++){
    for(int c=0; c<valid_mask.cols;c++){
      auto& color = rgb.at<cv::Vec3b>(r,c);
      if(valid_mask.at<uchar>(r,c))
        continue;
      for(int k=0; k<3; k++){
        color[k] = std::max<int>(0, (int)color[k]-diff);
      }
    }
  }
  return;
}

cv::Mat EntireVisualization(const cv::Mat _rgb,
                            const cv::Mat valid_depth,
                            const cv::Mat outline_edges,
                            const std::map<int, ShapePtr>& global_shapes,
                            const cv::Mat local_marker,
                            const std::vector<cv::KeyPoint>& keypoints
                            ) {
  cv::Mat rgb = _rgb.clone();
  HighlightValidmask(valid_depth, rgb);
  cv::Mat vis_edges_marker;
  cv::addWeighted(rgb, 1.,
                  GetColoredLabel(local_marker), .5,
                  1., vis_edges_marker);

  cv::Mat vis_trackedshapes;
  cv::Mat colored_shapes = rgb.clone();

  const int fontFace = cv::FONT_HERSHEY_SIMPLEX;
  const double fontScale = .3;
  const int fontThick = 1;

  std::set<int> draw_contours;
  for(auto it : global_shapes){
    ShapePtr ptr = it.second;
    if(!ptr->stabilized_)
      continue;
    if(ptr->n_missing_ > 0) // TODO?
      continue;
    draw_contours.insert(it.first);
    std::vector< std::vector<cv::Point> > cnts;
    cnts.resize(1);
    cnts[0].reserve(ptr->outerior_.size() );
    for( auto pt: ptr->outerior_)
      cnts[0].push_back(cv::Point(pt.x,pt.y));
    const auto& color = colors.at(it.first % colors.size() );
    cv::drawContours(colored_shapes, cnts, 0, color, -1);
  }

  cv::addWeighted(rgb, 1.,
                  colored_shapes, .5,
                  1., vis_trackedshapes);
  for(auto it : global_shapes){
    ShapePtr ptr = it.second;
    if(!draw_contours.count(it.first) )
      continue;
    std::vector< std::vector<cv::Point> > cnts;
    cnts.resize(1);
    cnts[0].reserve(ptr->outerior_.size() );
    for( auto pt: ptr->outerior_)
      cnts[0].push_back(cv::Point(pt.x,pt.y));
    const auto& color0 = colors.at(it.first % colors.size() );

    const auto& bb = ptr->outerior_bb_;
    const std::string txt = "#" + std::to_string(it.first);
    int baseline=0;
    auto size = cv::getTextSize(txt, fontFace, fontScale, fontThick, &baseline);
    cv::Point cp(bb.x+.5*bb.width, bb.y+.5*bb.height);
    cv::Point dpt(.5*size.width, .5*size.height);
    cv::rectangle(vis_trackedshapes,
                  cp-dpt-cv::Point(0,3*baseline),
                  cp+dpt,
                  CV_RGB(255,255,255), -1);
    cv::putText(vis_trackedshapes, txt, cp-dpt,fontFace, fontScale,
                color0, fontThick);
    cv::drawContours(vis_trackedshapes, cnts, 0, color0, 2);
  }

  bool show_keypoints = true;
  if(show_keypoints){
    for(const auto& kpt : keypoints) 
      cv::circle(vis_trackedshapes, kpt.pt, 3, CV_RGB(255,0,0), 1);
  }

  for(int r=0; r<rgb.rows; r++){
    for(int c=0; c<rgb.cols; c++){
      if( !outline_edges.at<uchar>(r,c) )
        continue;
      auto& color = vis_edges_marker.at<cv::Vec3b>(r,c);
      color[0] = 0;
      color[1] = 0;
      color[2] = 255;
    }
  }

  cv::pyrDown(vis_edges_marker,vis_edges_marker);

  cv::Mat dst = cv::Mat::zeros(std::max<int>(vis_edges_marker.rows, vis_trackedshapes.rows),
                               vis_edges_marker.cols + vis_trackedshapes.cols,
                               CV_8UC3);
  {
    cv::Rect rect(0, 0, vis_edges_marker.cols, vis_edges_marker.rows);
    cv::Mat roi(dst, rect);
    vis_edges_marker.copyTo(roi);
    cv::rectangle(dst, rect, CV_RGB(0,255,0), 2);
  }
  {
    cv::Rect rect(vis_edges_marker.cols,0,
                  vis_trackedshapes.cols, vis_trackedshapes.rows);
    cv::Mat roi(dst, rect);
    vis_trackedshapes.copyTo(roi);
    cv::rectangle(dst, rect, CV_RGB(0,255,0), 2);
  }
  return dst;
}

void Extract(cv::InputArray gray, const std::map<int, ShapePtr>& global_shapes,
             std::vector<cv::KeyPoint>& keypoints,
             cv::OutputArray descriptions) {
  /*
  # 기존 알고리즘
    기존 ORB SLAM의 keypoint (orb_extractor.cpp)는
    ORBextrctor::extract()
      1) ComputeKeyPointsOctTree()
        각 image pyramid level에서 >
          가로세로 30x30pixel과 짜투리 셀에 대해서,
          조각난 이미지를 FAST에 입력, level별로 분리해서 vector<vector<keypoint>에, 해당 lv scale의 uv 좌표 수집.
          keypoint를 분배하고, 그중 response가 강한것만 남기기위해,
          DistributeOctTree(), 가 keypoint를 Quadtree에 분배.
            각 노드는 포함하는 점이 1개만 남거나, node의 숫자가 required features보다 많아질때 중단된다.
            마지막으로 각 노드에서 가장 큰 response를 가진 keypoint만 추려낸다.
            즉, 화면 골고루에 강한점부터 뿌리고 점점 약한점이 사이사이에 추가된다.
      2) le:hi coc_err_hi ctermfg=1 ctermbg=15vel별 keypoints에 대해, scaled image를 사용, ORB descriptor를 추출하고, 
         scale을 복원해 std::vector<Keypoint>, cv::Mat descriptor 취합.
  # 개선,적용
    * 추출된 FAST point 중,
    Octree(lv별 Quadtree) Node별로, 가장 response가 강한 point부터 남기면서,
    이미 충분한 keypoint를 가진 shape에 위치한 keypoint는 skip < 이기능만 DistributeOctree 단계에서 추가.
       TODO) Inprogress, 맞다. keypoint extractor 구현.
    * 우려사항 : FAST가 각 shape를 추적하기에 충분할 만큼 feature point 추출이 가능한가?
      * 차라리 shape BB 안에 한해서 다시 FAST를 따로 돌리는게 났다? ... 이건 느리기만하니 패스.
      * 모든 segment에 대해 상대 '좌표계' - 특히 orientation을 계산하는건 어렵다.
       - feature point가 충분히 추적안되는 instnace도 '분리'는 필요.
       - EdgeSE3Switchable 대신에, 무엇을 error로 삼아야할까?
  */
   
  // ORBextractor::extract()
  int nfeatures = 2000;
  float scale_factor = 1.2;
  int nlevels = 8;
  int initial_fast_th = 20;
  int min_fast_th = 7;
  static std::unique_ptr<ORB_SLAM2::ORBextractor> extractor(new ORB_SLAM2::ORBextractor(nfeatures, scale_factor,
                                                                  nlevels, initial_fast_th, min_fast_th));
  extractor->extract(gray, cv::noArray(), keypoints, descriptions);

  return;
}

void Seg::_Put(cv::Mat gray,
               cv::cuda::GpuMat g_gray,
               cv::Mat depth,
               const Camera& camera,
               cv::Mat rgb
               ) {
  auto start = std::chrono::steady_clock::now();
  std::vector<cv::Point2f> corners = GetCorners(gray);
  if(corners.empty())
    return;
  if(gray0_.empty()){
    PutKeyframe(gray, g_gray);
    return;
  }
  const bool is_keyframe = true;
  cv::Mat flow = GetFlow(g_gray); // Flow for tracking shape.
#if 1
#else
  // Flow대신 ORB matcher 적용
  IsKeyframe(flow,rgb);
#endif
  cv::Mat valid_depth = depth > 0.;
  cv::Mat filtered_depth;
  cv::erode(valid_depth,valid_depth,cv::Mat::ones(13,13,CV_32FC1) );
  cv::GaussianBlur(depth,filtered_depth, cv::Size(7,7), 0., 0.); // Shallow groove가 필요없어서 그냥 Gaussian

  cv::Mat gradx, grady, valid_grad;
  cv::Mat vis_grad = GetGrad(filtered_depth, valid_depth, camera, gradx, grady, valid_grad, true);
  cv::Mat valid;
  cv::bitwise_and(valid_depth, valid_grad, valid);
  cv::Mat concave_edges = GetConcaveEdges(gradx,grady,depth,valid,camera);
  //concave_edges = FilterThinNoise(concave_edges);

  cv::Mat dd_edges = GetDDEdges(filtered_depth, valid_depth, camera); // concave_edge를 보완해주는 positive detection이 없음.
  cv::Mat outline_edges;
  cv::bitwise_or(concave_edges, dd_edges, outline_edges);
  outline_edges = FilterThinNoise(outline_edges);

  bool limit_expand_range = false;
  cv::Mat marker = Segment(outline_edges,valid_depth,limit_expand_range); // Sgement(error_edges,rgb);
 /* 
  * [x] Visualzation 개선 / 후 commit
  * [x] keypoint extraction, visualization
  * [ ] Switchable edge 기반 LBA
 */
  std::map<int, ShapePtr> local_shapes = ConvertMarker2Instances(marker);
  static std::map<int, ShapePtr> global_shapes; // TODO 맴버변수로
  static int n_shapes = 0;
  const float min_iou = .3;
  std::map<int,int> l2g = TrackShapes(local_shapes, marker, flow, min_iou, global_shapes, n_shapes);

  std::vector<cv::KeyPoint> keypoints;
  cv::Mat descriptions;
  Extract(gray, global_shapes, keypoints, descriptions);
  // TODO LBA

  if(!is_keyframe){
    cv::waitKey(1);
    return;
  }

  if(is_keyframe)
    PutKeyframe(gray, g_gray);

  auto stop = std::chrono::steady_clock::now();
  std::cout << "etime = " << std::chrono::duration_cast<std::chrono::milliseconds>(stop-start).count() << "[msec]" << std::endl;
  {
    cv::Mat dst = EntireVisualization(rgb, valid_depth, outline_edges, global_shapes, marker,
                                      keypoints);
    cv::imshow("vis", dst);
  }
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
