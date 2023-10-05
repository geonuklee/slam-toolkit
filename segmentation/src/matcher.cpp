#include "pybind11/attr.h"
#include "segslam.h"
#include "camera.h"

inline double lerp(double x0, double y0, double x1, double y1, double x) {
  return y0 + (x - x0) * (y1 - y0) / (x1 - x0);
}

namespace NEW_SEG {
std::map<int, std::pair<Mappoint*, double> > FlowMatch(const Camera* camera,
                                                       const SEG::FeatureDescriptor* extractor,
                                                       const std::vector<cv::Mat>& flow,
                                                       const Frame* prev_frame,
                                                       bool verbose,
                                                       Frame* curr_frame) {
  double search_threshold = 2.;
  double best12_threshold = .8;
  //const double search_radius_min = 5.; // [pixel]
  //const double far_depth         = 50.; // [meter] for min radius.
  //const double search_radius_max = 20.; // [pixel]
  //const double near_depth        = 20.; // [meter] for max radius
  const auto& keypoints0 = prev_frame->GetKeypoints();
  const auto& mappoints0 = prev_frame->GetMappoints();
  const auto& depths0    = prev_frame->GetMeasuredDepths();
  const auto& keypoints1 = curr_frame->GetKeypoints();
  std::map<int, std::pair<Mappoint*, double> > matches;
  for(size_t iter = 0; iter <2; iter++){
    std::vector<int> grids = {0, 0, 0, 0};
    cv::Mat dst0, dst1;
    if(verbose){
      dst0 = prev_frame->GetRgb().clone();
      dst1 = curr_frame->GetRgb().clone();
      for(int n=0; n<keypoints0.size(); n++)
        cv::circle(dst0,  keypoints0[n].pt, 3, CV_RGB(150,150,150), 1);
      for(int n=0; n<keypoints1.size(); n++)
        cv::circle(dst1,  keypoints1[n].pt, 3, CV_RGB(150,150,150), 1);
    }

    for(int n0=0; n0 < mappoints0.size(); n0++){
      Mappoint* mp0 = mappoints0[n0];
      if(!mp0)
        continue;
      const cv::KeyPoint& kpt0 = keypoints0[n0];
      const cv::Point2f&  pt0  = kpt0.pt;
      //const auto& dpt01 = flow0.at<cv::Point2f>(pt0);
      cv::Point2f dpt01(flow[0].at<float>(pt0), flow[1].at<float>(pt0));
      Eigen::Vector2d eig_pt1(pt0.x+dpt01.x, pt0.y+dpt01.y);
      if(!curr_frame->IsInFrame(camera,eig_pt1))  //  땅바닥 제외
        continue;

      if(verbose){
        cv::Point2f pt1(eig_pt1.x(), eig_pt1.y());
        cv::circle(dst0, pt0, 3,  CV_RGB(255,0,0), -1);
        cv::line(  dst0, pt0, pt1, CV_RGB(0,255,0), 1);
        cv::circle(dst1, pt1, 3,  CV_RGB(255,0,0), -1);
      }
      std::list< std::pair<int,double> > candidates = curr_frame->SearchRadius(eig_pt1, search_threshold);
      if(candidates.empty())
        continue;
      const cv::Mat desc0 = prev_frame->GetDescription(n0);
      double dist0 = 1e+9;
      double dist1 = dist0;
      int champ0 = -1;
      int champ1 = -1;
      float champ0_err = 0.;
      for(const auto& candi : candidates){
        const int& n1 = candi.first;
        const cv::Mat desc1 = curr_frame->GetDescription(n1);
        const cv::KeyPoint& kpt1 = keypoints1[n1];
        //if(std::abs(kpt0.octave-kpt1.octave) > 1) continue;
        //float diff_angle = std::abs(kpt0.angle-kpt1.angle);
        //if(diff_angle >  180) diff_angle -= 360; // cv::fastAtan2 degree.
        //if(diff_angle < -180) diff_angle += 360;
        double dist = extractor->GetDistance(desc0,desc1);
        if(dist < dist0) {
          dist1 = dist0;
          champ1 = champ0;
          dist0 = dist;
          champ0 = n1;
          champ0_err = candi.second;
        }
        else if(dist < dist1){
          dist1 = dist;
          champ1 = n1;
        }
      }
      if(champ0_err > search_threshold)
        continue;
      if(champ0 < 0)
        continue;
      /*{
      const cv::KeyPoint& kpt1 = keypoints1[champ0];
      if(std::abs(kpt0.angle - kpt1.angle) > 20.) // flow match는 낮은 angle 오차 허용.
      continue;
      if(kpt0.octave != kpt1.octave)
      continue;
      }*/
      if(dist0 < dist1 * best12_threshold){
        if(matches.count(champ0)){
          // 이미 matching candidate가 있는 keypoint가 선택된 경우,
          // 오차를 비교하고 교체 여부 결정
          if(matches.at(champ0).second < dist0)
            continue;
        }
        matches[champ0] = std::make_pair(mp0,dist0);
      }
    } // for mappoints0
    if(!dst1.empty()){ // Visualization
      for(auto it : matches){
        const int& i1 = it.first;
        const cv::Point2f& pt1 = curr_frame->GetKeypoint(i1).pt;
        const int& i0 = prev_frame->GetIndex(it.second.first);
        const cv::Point2f& pt0 = prev_frame->GetKeypoint(i0).pt;
        cv::circle(dst0, pt0, 3,   CV_RGB(0,0,255), -1);
        cv::circle(dst1, pt1, 3,   CV_RGB(0,0,255), -1);
        cv::line(  dst1, pt1, pt0, CV_RGB(0,255,0), 1);
      }
      cv::imshow("flow0", dst0);
      cv::imshow("flow1", dst1);
    }
#if 0
    // matches의 분포로 재연결 판정.
    for(auto it : matches){
      const cv::KeyPoint& kpt1 = curr_frame->GetKeypoint(it.first);
      int r = int( grids.size() * (kpt1.pt.x / camera->GetWidth()) );
      r = std::max<int>(0,r);
      r = std::min<int>(grids.size()-1,r);
      grids[r]++;
    }
    if(grids[0]+grids[1]> 10 && grids[2]+grids[3]>10)
      break;
    std::cout << "Warning : F#" << curr_frame->GetId() << ", difficult to flow matches" << std::endl;
    matches.clear();
    best12_threshold = 2.;
    search_threshold = 5.;
    search_threshold = 5.;
#else
    break;
#endif
  } // for iter
  return matches;
} // NEW_SEG::FlowMatch

std::map<int, std::pair<Mappoint*,double> > ProjectionMatch(const Camera* camera,
                                                            const SEG::FeatureDescriptor* extractor,
                                                            const std::set<Mappoint*>& mappoints,
                                                            const Frame* curr_frame, // With predicted Tcq
                                                            const Qth qth
                                                            ) {
  const double search_radius = 15.;
  const double search_radius2 = 2.*search_radius;
  const double best12_threshold = .5;

  const bool use_latest_desc = false;
  double* ptr_projections = new double[2 * mappoints.size()];
  std::map<int, Mappoint*> key_table;
  int n = 0;
  for(Mappoint* mp : mappoints){
    if(curr_frame->GetIndex(mp) >= 0){
      // flow match에서 연결된경우.
      // Loopclosing이 2번 이상 한 지점에서 발생하면
      // LoopCloser::CombineNeighborMappoints()에서 발생하는 경우.
      continue;
    }
    const Eigen::Vector3d Xq = mp->GetXq(qth);
    const Eigen::Vector3d Xc = curr_frame->GetTcq(qth) * Xq;
    if(Xc.z() < 0.)
      continue;
    Eigen::Vector2d uv = camera->Project(Xc);
    if(!curr_frame->IsInFrame(camera,uv))  //  땅바닥 제외
      continue;
    ptr_projections[2*n]   = uv[0];
    ptr_projections[2*n+1] = uv[1];
    key_table[n++] = mp;
  }

  std::map<int, std::pair<Mappoint*,double> > matches;
  flann::Matrix<double> queries(ptr_projections, n, 2);
  std::vector<std::vector<double> > err_squares;
  std::vector<std::vector<int> > batch_search = curr_frame->SearchRadius(queries, search_radius2, err_squares);

  for(size_t key_mp = 0; key_mp < batch_search.size(); key_mp++){
    Mappoint* query_mp = key_table.at(key_mp);
    cv::Mat desc0; cv::KeyPoint kpt0;
    query_mp->GetFeature(qth, use_latest_desc, desc0, kpt0);
    const std::vector<int>& each_search = batch_search.at(key_mp);
    const std::vector<double>& each_err_square = err_squares.at(key_mp);
    double dist0 = 999999999.;
    double dist1 = dist0;
    int champ0 = -1;
    int champ1 = -1;
    double champ0_err = 0.;
    for(int n=0; n < each_search.size(); n++){
      int idx = each_search.at(n);
      double err_square = each_err_square.at(n);
      cv::Mat desc1 = curr_frame->GetDescription(idx);
      const cv::KeyPoint& kpt1 = curr_frame->GetKeypoint(idx);
      if(! curr_frame->IsInFrame(camera, Eigen::Vector2d(kpt1.pt.x, kpt1.pt.y)) )
        continue;
      double dist = extractor->GetDistance(desc0,desc1);
      //if(std::abs(kpt0.angle - kpt1.angle) > 40.) continue
      if(std::abs(kpt0.octave-kpt1.octave) > 1)
        continue;
      if(dist < dist0){
        dist1 = dist0;
        champ1 = champ0;
        dist0 = dist;
        champ0 = idx;
        champ0_err = std::sqrt(err_square);
      }
      else if(dist < dist1){
        dist1 = dist;
        champ1 = idx;
      }
    }
    if(champ0 < 0)
      continue;
    if(champ0_err > search_radius)
      continue;
    if(dist0 < dist1 * best12_threshold){
      if(matches.count(champ0)){
        // 이미 matching candidate가 있는 keypoint가 선택된 경우,
        // 오차를 비교하고 교체 여부 결정
        if(matches.at(champ0).second < dist0)
          continue;
      }
      matches[champ0] = std::make_pair(query_mp, dist0);
    }
  }
  delete[] ptr_projections;
  return matches;
}


} // namespace NEW_SEG
