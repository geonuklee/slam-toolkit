#include "pybind11/attr.h"
#include "segslam.h"

inline double lerp(double x0, double y0, double x1, double y1, double x) {
  return y0 + (x - x0) * (y1 - y0) / (x1 - x0);
}

namespace OLD_SEG {
std::map<int, std::pair<Mappoint*, double> > FlowMatch(const Camera* camera,
                                                       const SEG::FeatureDescriptor* extractor,
                                                       const std::vector<cv::Mat>& flow,
                                                       const Frame* prev_frame,
                                                       bool verbose,
                                                       Frame* curr_frame) {
  const double best12_threshold = .7;
  const double search_radius_min = 5.; // [pixel]
  const double far_depth         = 50.; // [meter] for min radius.
  const double search_radius_max = 20.; // [pixel]
  const double near_depth        = 15.; // [meter] for max radius
  const auto& keypoints0 = prev_frame->GetKeypoints();
  const auto& mappoints0 = prev_frame->GetMappoints();
  const auto& depths0    = prev_frame->GetMeasuredDepths();
  const auto& keypoints1 = curr_frame->GetKeypoints();
  cv::Mat dst0, dst1;
  if(verbose){
    dst0 = prev_frame->GetRgb().clone();
    dst1 = curr_frame->GetRgb().clone();
    for(int n=0; n<keypoints0.size(); n++)
      cv::circle(dst0,  keypoints0[n].pt, 3, CV_RGB(150,150,150), 1);
    for(int n=0; n<keypoints1.size(); n++)
      cv::circle(dst1,  keypoints1[n].pt, 3, CV_RGB(150,150,150), 1);
  }
  std::map<int, std::pair<Mappoint*, double> > matches;
  for(int n0=0; n0 < mappoints0.size(); n0++){
    Mappoint* mp0 = mappoints0[n0];
    if(!mp0)
      continue;
    const cv::KeyPoint& kpt0 = keypoints0[n0];
    const cv::Point2f&  pt0  = kpt0.pt;
    const double z = depths0[n0] < 1e-5 ? far_depth : depths0[n0];
    double search_radius = lerp(near_depth, search_radius_max, far_depth, search_radius_min, z);
    search_radius = std::max(search_radius, search_radius_min);
    search_radius = std::min(search_radius, search_radius_max);
    //const auto& dpt01 = flow0.at<cv::Point2f>(pt0);
    cv::Point2f dpt01(flow[0].at<float>(pt0.x), flow[1].at<float>(pt0.y));
    Eigen::Vector2d eig_pt1(pt0.x+dpt01.x, pt0.y+dpt01.y);
    if(!curr_frame->IsInFrame(camera,eig_pt1))  //  땅바닥 제외
      continue;

    if(verbose){
      cv::Point2f pt1(eig_pt1.x(), eig_pt1.y());
      cv::line(  dst1, pt0, pt1, CV_RGB(0,0,255), 1);
      cv::circle(dst1, pt1, 3,  CV_RGB(0,0,255), -1);
      cv::line(  dst0, pt0, pt1, CV_RGB(0,0,255), 1);
      cv::circle(dst0, pt0, 3,  CV_RGB(0,0,255), -1);
      char text[100];
      sprintf(text,"%2.1f",z);
      cv::putText(dst0, text, pt0, cv::FONT_HERSHEY_SIMPLEX, .4,  CV_RGB(255,0,0) );
      cv::circle(dst0, pt1, search_radius,  CV_RGB(0,255,0), 1);
    }

    std::list<int> candidates = curr_frame->SearchRadius(eig_pt1, search_radius);
    if(candidates.empty())
      continue;
    const cv::Mat desc0 = prev_frame->GetDescription(n0);
    double dist0 = 1e+9;
    double dist1 = dist0;
    int champ0 = -1;
    int champ1 = -1;
    for(const int& n1 : candidates){
      const cv::Mat desc1 = curr_frame->GetDescription(n1);
      const cv::KeyPoint& kpt1 = keypoints1[n1];
      double dist = extractor->GetDistance(desc0,desc1);
      if(dist < dist0){
        dist1 = dist0;
        champ1 = champ0;
        dist0 = dist;
        champ0 = n1;
      }
      else if(dist < dist1){
        dist1 = dist;
        champ1 = n1;
      }
    }
    if(champ0 < 0)
      continue;
    {
      const cv::KeyPoint& kpt1 = keypoints1[champ0];
      if(std::abs(kpt0.angle - kpt1.angle) > 20.) // flow match는 낮은 angle 오차 허용.
        continue;
      if(kpt0.octave != kpt1.octave)
        continue;
    }
    bool c2 = dist0 < dist1 * best12_threshold;
    if(candidates.size() < 6 || c2  ){
      if(matches.count(champ0)){
        // 이미 matching candidate가 있는 keypoint가 선택된 경우,
        // 오차를 비교하고 교체 여부 결정
        if(matches.at(champ0).second < dist0)
          continue;
      }
      matches[champ0] = std::make_pair(mp0,dist0);
    }
  }

  if(!dst1.empty()){ // Visualization
    for(auto it : matches){
      const int& i1 = it.first;
      const cv::Point2f& pt1 = curr_frame->GetKeypoint(i1).pt;
      const int& i0 = prev_frame->GetIndex(it.second.first);
      const cv::Point2f& pt0 = prev_frame->GetKeypoint(i0).pt;
      cv::line(  dst1, pt0, pt1, CV_RGB(255,255,0), 1);
      cv::circle(dst1, pt1, 3,  CV_RGB(255,255,0), 1);
    }
    char buffer[200];
    snprintf(buffer, sizeof(buffer), "Jth #%d, #%d", prev_frame->GetId(), curr_frame->GetId() );
    cv::putText(dst1, buffer, cv::Point(10, dst1.rows - 20), cv::FONT_HERSHEY_SIMPLEX, 1.,  CV_RGB(255,0,0), 2);
    cv::imshow("flow 01", dst0);
    cv::imshow("flow 10", dst1);
  }
  return matches;
}
} // namespace OLD_SEG

namespace NEW_SEG {
std::map<int, std::pair<Mappoint*, double> > FlowMatch(const Camera* camera,
                                                       const SEG::FeatureDescriptor* extractor,
                                                       const std::vector<cv::Mat>& flow,
                                                       const Frame* prev_frame,
                                                       bool verbose,
                                                       Frame* curr_frame) {
  const double best12_threshold = .7;
  const double search_radius_min = 5.; // [pixel]
  const double far_depth         = 50.; // [meter] for min radius.
  const double search_radius_max = 20.; // [pixel]
  const double near_depth        = 20.; // [meter] for max radius
  const auto& keypoints0 = prev_frame->GetKeypoints();
  const auto& mappoints0 = prev_frame->GetMappoints();
  const auto& depths0    = prev_frame->GetMeasuredDepths();
  const auto& keypoints1 = curr_frame->GetKeypoints();
  cv::Mat dst0, dst1;
  if(verbose){
    dst0 = prev_frame->GetRgb().clone();
    dst1 = curr_frame->GetRgb().clone();
    for(int n=0; n<keypoints0.size(); n++)
      cv::circle(dst0,  keypoints0[n].pt, 3, CV_RGB(150,150,150), 1);
    for(int n=0; n<keypoints1.size(); n++)
      cv::circle(dst1,  keypoints1[n].pt, 3, CV_RGB(150,150,150), 1);
  }
  std::map<int, std::pair<Mappoint*, double> > matches;
  for(int n0=0; n0 < mappoints0.size(); n0++){
    Mappoint* mp0 = mappoints0[n0];
    if(!mp0)
      continue;
    const cv::KeyPoint& kpt0 = keypoints0[n0];
    const cv::Point2f&  pt0  = kpt0.pt;
    const double z = depths0[n0] < 1e-5 ? far_depth : depths0[n0];
#if 1
    double search_radius = lerp(near_depth, search_radius_max, far_depth, search_radius_min, z);
    search_radius = std::max(search_radius, search_radius_min);
    search_radius = std::min(search_radius, search_radius_max);
#else
    double search_radius = 5.;
#endif
    //const auto& dpt01 = flow0.at<cv::Point2f>(pt0);
    cv::Point2f dpt01(flow[0].at<float>(pt0.x), flow[1].at<float>(pt0.y));
    Eigen::Vector2d eig_pt1(pt0.x+dpt01.x, pt0.y+dpt01.y);
    if(!curr_frame->IsInFrame(camera,eig_pt1))  //  땅바닥 제외
      continue;

    if(verbose){
      cv::Point2f pt1(eig_pt1.x(), eig_pt1.y());
      cv::line(  dst1, pt0, pt1, CV_RGB(0,0,255), 1);
      cv::circle(dst1, pt1, 3,  CV_RGB(0,0,255), -1);
      cv::line(  dst0, pt0, pt1, CV_RGB(0,0,255), 1);
      cv::circle(dst0, pt0, 3,  CV_RGB(0,0,255), -1);
      char text[100];
      sprintf(text,"%2.1f",z);
      cv::putText(dst0, text, pt0, cv::FONT_HERSHEY_SIMPLEX, .4,  CV_RGB(255,0,0) );
      cv::circle(dst0, pt1, search_radius,  CV_RGB(0,255,0), 1);
    }

    std::list<int> candidates = curr_frame->SearchRadius(eig_pt1, search_radius);
    if(candidates.empty())
      continue;
    const cv::Mat desc0 = prev_frame->GetDescription(n0);
    double dist0 = 1e+9;
    double dist1 = dist0;
    int champ0 = -1;
    int champ1 = -1;
    for(const int& n1 : candidates){
      const cv::Mat desc1 = curr_frame->GetDescription(n1);
      const cv::KeyPoint& kpt1 = keypoints1[n1];
      double dist = extractor->GetDistance(desc0,desc1);
      if(dist < dist0){
        dist1 = dist0;
        champ1 = champ0;
        dist0 = dist;
        champ0 = n1;
      }
      else if(dist < dist1){
        dist1 = dist;
        champ1 = n1;
      }
    }
    if(champ0 < 0)
      continue;
    {
      const cv::KeyPoint& kpt1 = keypoints1[champ0];
      if(std::abs(kpt0.angle - kpt1.angle) > 20.) // flow match는 낮은 angle 오차 허용.
        continue;
      if(kpt0.octave != kpt1.octave)
        continue;
    }
    bool c2 = dist0 < dist1 * best12_threshold;
    if(candidates.size() < 6 || c2  ){
      if(matches.count(champ0)){
        // 이미 matching candidate가 있는 keypoint가 선택된 경우,
        // 오차를 비교하고 교체 여부 결정
        if(matches.at(champ0).second < dist0)
          continue;
      }
      matches[champ0] = std::make_pair(mp0,dist0);
    }
  }

  if(!dst1.empty()){ // Visualization
    for(auto it : matches){
      const int& i1 = it.first;
      const cv::Point2f& pt1 = curr_frame->GetKeypoint(i1).pt;
      const int& i0 = prev_frame->GetIndex(it.second.first);
      const cv::Point2f& pt0 = prev_frame->GetKeypoint(i0).pt;
      cv::line(  dst1, pt0, pt1, CV_RGB(255,255,0), 1);
      cv::circle(dst1, pt1, 3,  CV_RGB(255,255,0), 1);
    }
    char buffer[200];
    snprintf(buffer, sizeof(buffer), "Jth #%d, #%d", prev_frame->GetId(), curr_frame->GetId() );
    cv::putText(dst1, buffer, cv::Point(10, dst1.rows - 20), cv::FONT_HERSHEY_SIMPLEX, 1.,  CV_RGB(255,0,0), 2);
    cv::imshow("flow 01", dst0);
    cv::imshow("flow 10", dst1);
  }
  return matches;
}
} // namespace NEW_SEG
