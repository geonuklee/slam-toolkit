/*
Copyright (c) 2020 Geonuk Lee

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

#include "matcher.h"
#include "frame.h"
#include "mappoint.h"
#include "camera.h"
#include "orb_extractor.h"

std::set<int> SearchLine(cv::Point2f pt0,
                         cv::Point2f pt1,
                         double threshold,
                         const std::vector<cv::KeyPoint>& keypoints){
  Eigen::Vector2d uv0(pt0.x, pt0.y);
  Eigen::Vector2d uv1(pt1.x, pt1.y);
  Eigen::Vector3d uv10;
  uv10.head<2>() = (uv1 - uv0).normalized();
  uv10[2] = 0.;

  std::set<int> inlier;
  for(size_t i = 0; i < keypoints.size(); i++){
    auto kpt = keypoints.at(i);
    Eigen::Vector3d p;
    p.head<2>() = Eigen::Vector2d(kpt.pt.x, kpt.pt.y) - uv0;
    p[2] = 0.;
    double error =  std::abs(uv10.cross(p)[2]);
    if(error > threshold)
      continue;
    inlier.insert(i);
  }
  return inlier;
}

void StereoMatch(StereoFrame* frame){
  const auto& keypoints = frame->GetKeypoints();
  auto mappoints = frame->GetVecMappoints(); // StereoMatch
  const auto& r_keypoints = frame->GetRightKeypoints();
  size_t width = frame->GetCamera()->GetWidth();

  double step_size = 10;  // serach from upper, lower.
  std::map<int, std::set<int> > row_indices;
  for(size_t j = 0; j < r_keypoints.size(); j++){
    cv::KeyPoint rkpt = r_keypoints.at(j);
    int idx = (int) (rkpt.pt.y / step_size);
    row_indices[idx].insert(j);
  }

  double y_threshold = 3.;
  double best12_threshold = 0.5;
  double max_dx = 100.;

  std::vector<int> stereo_indices;
  stereo_indices.resize(keypoints.size(), -1);

  for(size_t i = 0; i < keypoints.size(); i++) {
    //Mappoint* mp = mappoints.at(i);
    //if(mp)
    //  continue;
    cv::KeyPoint kpt = keypoints.at(i);
    cv::Point2f pt0(0., kpt.pt.y), pt1(width, kpt.pt.y);
    cv::Mat idesc = frame->GetDescription(i);

    std::set<int> inliers;
    {
      int row = kpt.pt.y/step_size;
      inliers.insert(row_indices[row].begin(), row_indices[row].end());
    }
    {
      int row = kpt.pt.y/step_size-1;
      inliers.insert(row_indices[row].begin(), row_indices[row].end());
    }
    {
      int row = kpt.pt.y/step_size+1;
      inliers.insert(row_indices[row].begin(), row_indices[row].end());
    }

    double dist0 = 999999999.;
    double dist1 = dist0;
    int champ0 = -1;
    int champ1 = -1;
    for(int j : inliers){
      cv::KeyPoint rkpt = r_keypoints.at(j);
      double dx = kpt.pt.x - rkpt.pt.x ;
      double dy = kpt.pt.y - rkpt.pt.y ;
      if(std::abs(dy) > y_threshold)
        continue;
      if(dx < 0.)
        continue;
      if(dx > max_dx)
        continue;

      cv::Mat jdesc = frame->GetRightDescription(j);
      double dist = ORB_SLAM2::ORBextractor::DescriptorDistance(idesc, jdesc);
      if(dist < dist0){
        dist1 = dist0;
        champ1 = champ0;
        dist0 = dist;
        champ0 = j;
      }
      else if(dist < dist1){
        dist1 = dist;
        champ1 = j;
      }
    }
    if(champ0 < 0)
      continue;
    if(dist0 < dist1 * best12_threshold)
      stereo_indices[i] = champ0;
  }
  frame->SetStereoCorrespond(stereo_indices);
  return;
}

std::map<int, Mappoint*> ProjectionMatch(const std::set<Mappoint*>& mappoints,
                                         const g2o::SE3Quat& predicted_Tcw,
                                         const Frame* curr_frame,
                                         double search_radius) {
  const double best12_threshold = 0.5;
  const Camera* camera = curr_frame->GetCamera();
  double* ptr_projections = new double[2 * mappoints.size()];
  std::map<int, Mappoint*> key_table_;
  int n = 0;
  for(Mappoint* mp : mappoints){
    if(curr_frame->GetIndex(mp) >= 0){
      // Loopclosing이 2번 이상 한 지점에서 발생하면
      // LoopCloser::CombineNeighborMappoints()에서 발생하는 경우.
      continue;
    }

    Eigen::Vector3d Xw = mp->GetXw();
    Eigen::Vector3d Xc = predicted_Tcw * Xw;
    if(Xc.z() < 0.)
      continue;
    Eigen::Vector2d uv = camera->Project(Xc);
    if(!curr_frame->IsInFrame(uv))
      continue;
    ptr_projections[2*n]   = uv[0];
    ptr_projections[2*n+1] = uv[1];
    key_table_[n++] = mp;
  }

  flann::Matrix<double> quaries(ptr_projections, n, 2);
  std::vector<std::vector<int> > batch_search = curr_frame->SearchRadius(quaries, search_radius);

  std::map<int, Mappoint*> matches;
  std::map<int, double> distances;
  for(size_t key_mp = 0; key_mp < batch_search.size(); key_mp++){
    Mappoint* quary_mp = key_table_.at(key_mp);
    cv::Mat desc0 = quary_mp->GetDescription();
    const std::vector<int>& each_search = batch_search.at(key_mp);
    double dist0 = 999999999.;
    double dist1 = dist0;
    int champ0 = -1;
    int champ1 = -1;
    for(int idx : each_search){
      cv::Mat desc1 = curr_frame->GetDescription(idx);
      double dist = ORB_SLAM2::ORBextractor::DescriptorDistance(desc0, desc1);
      //if(std::abs(kpt0.angle - kpt.angle) > 40.)
      //  continue;
      //if(kpt0.octave != kpt.octave)
      //  continue;
      if(dist < dist0){
        dist1 = dist0;
        champ1 = champ0;
        dist0 = dist;
        champ0 = idx;
      }
      else if(dist < dist1){
        dist1 = dist;
        champ1 = idx;
      }
    }

    if(champ0 < 0)
      continue;
    if(dist0 < dist1 * best12_threshold){
      if(matches.count(champ0)){
        // 이미 matching candidate가 있는 keypoint가 선택된 경우,
        // 오차를 비교하고 교체 여부 결정
        if(distances.at(champ0) < dist0)
          continue;
      }
      matches[champ0] = quary_mp;
      distances[champ0] = dist0;
    }
  }
  delete[] ptr_projections;
  return matches;
}

