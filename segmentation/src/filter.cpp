#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include <vector>
#include <cstdlib>  // for srand

#include "segslam.h"
#include "util.h"
#include "optimizer.h"

namespace NEW_SEG {
std::set<Pth> Pipeline::NewFilterOutlierMatches(Frame* curr_frame,const EigenMap<Qth, g2o::SE3Quat>& Tcps, bool verbose) {
  const double focal = camera_->GetK()(0,0);
  bool use_extrinsic_guess = true;
  double confidence = .99;
  cv::Mat& dst = vinfo_match_filter_;
  if(verbose){
    dst = GetColoredLabel(vinfo_synced_marker_);
    dst.setTo(CV_RGB(255,0,0), GetBoundary(vinfo_synced_marker_,2));
    cv::addWeighted(cv::Mat::zeros(dst.rows,dst.cols,CV_8UC3), .5, dst, .5, 1., dst);
  }
  int flag = cv::SOLVEPNP_ITERATIVE;
  cv::Mat K = cvt2cvMat(camera_->GetK() );
  cv::Mat D = cvt2cvMat(camera_->GetD() );
  const std::vector<Mappoint*>&    _mappoints = curr_frame->GetMappoints();
  struct Points {
    cv::Point3f pt3d_prev;
    cv::Point3f uvz_curr;
    int kptid_curr;
    int kptid_prev;
    Mappoint* mp;
  };
  std::map<Instance*,std::list<Points> > segmented_points;
  for(int n =0; n < _mappoints.size(); n++){
    Points pt;
    pt.mp = _mappoints.at(n);
    pt.kptid_curr = n;
    if(!pt.mp)
      continue;
    pt.kptid_prev = prev_frame_->GetIndex(pt.mp);
    if(pt.kptid_prev < 0)
      continue;
    Instance* ins = pt.mp->GetLatestInstance();
    if(!ins)
      continue;
    const Eigen::Vector3d Xp = prev_frame_->GetDepth(pt.kptid_prev) * prev_frame_->GetNormalizedPoint(pt.kptid_prev);
    pt.pt3d_prev.x = Xp.x(); pt.pt3d_prev.y = Xp.y(); pt.pt3d_prev.z = Xp.z();
    const auto& nuv = curr_frame->GetNormalizedPoint(pt.kptid_curr).head<2>();
    pt.uvz_curr.x = nuv.x();
    pt.uvz_curr.y = nuv.y();
    pt.uvz_curr.z = curr_frame->GetDepth(pt.kptid_curr);
    segmented_points[ins].push_back(pt);
  }

  std::vector<cv::Point3f> vec_prev, vec_uvz_curr;
  vec_uvz_curr.reserve(_mappoints.size());
  vec_prev.reserve(_mappoints.size());
  std::set<Pth> pnp_fixed_instances;

  const StereoCamera* scam = dynamic_cast<const StereoCamera*>(camera_);
  double uv_std = 1. / focal; // normalized standard deviation
  const double uv_info = 1./uv_std/uv_std;
  const auto Trl_ = scam->GetTrl();
  const float base_line = -Trl_.translation().x();
  const double invd_info = uv_info * base_line * base_line;  // focal legnth 1.인 normalized image ponint임을 고려.
  const double delta = 5./focal;

  // TODO Fisher information에 의해 nomralized된 3D 3D를 비교해야 한다. translation의 scale은 필요 없음.
  const double th2d = ChiSquaredThreshold(.7, 2); // seq 05, frame 180 작고 빠른, 맞은편 instance에 트래킹 유지정도 되려면.., uvz
  const double th3d = ChiSquaredThreshold(.7, 3);
  //double rprj_threshold = 3./focal;
  //const double th2d = 2.*rprj_threshold*rprj_threshold*uv_info;
  //const double th3d = th2d + .5*th2d*base_line*base_line;

  for(auto it : segmented_points){
    if(it.second.size() < 10){ // TODO set min num points
      pnp_fixed_instances.insert(it.first->GetId());
      continue;
    }
    vec_uvz_curr.clear();
    vec_prev.clear();
    for(const auto& it_pt : it.second){
      vec_prev.push_back(it_pt.pt3d_prev);
      vec_uvz_curr.push_back(it_pt.uvz_curr);
    }
    // PnPRANSAC이 적합하지 않으니, optimizer에서 Tcp 를 찾고, residual chi2 > 95%인 outlier를 제거.
    std::vector<double> vec_chi2;
    g2o::SE3Quat Tcp = EstimateTcp(vec_prev, vec_uvz_curr, camera_, uv_info, invd_info, delta, vec_chi2);
    auto it_points = it.second.begin();
    for(int i=0; i < vec_prev.size(); i++){
      const auto& uvz = vec_uvz_curr.at(i);
      bool valid_depth = uvz.z > 1e-5;
      bool inlier = vec_chi2.at(i) < (valid_depth ? th3d : th2d);
      const Points& pt = *it_points;
      if(verbose){
        const auto& kpt = curr_frame->GetKeypoint(pt.kptid_curr).pt;
        cv::circle(dst, kpt, 3, inlier?CV_RGB(0,255,0):CV_RGB(255,0,0), -1 );
      }
      if(!inlier){
        curr_frame->EraseMappoint(pt.kptid_curr); // keyframe이 아니라서 mp->RemoveKeyframe 등을 호출하지 않는다.
        if(prev_frame_->IsKeyframe())
          pt.mp->RemoveKeyframe(prev_frame_);
        prev_frame_->EraseMappoint(pt.kptid_prev); // keyframe에서..
      }
      it_points++;
    }
  }
  return pnp_fixed_instances;
}

} // namespace NEW_SEG


