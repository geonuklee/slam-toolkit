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

#ifndef FRAME_H_
#define FRAME_H_
#include <flann/flann.hpp> // include it before opencv

#include "stdafx.h"
#include "common.h"
#include "ORBVocabulary.h"

#include "thirdparty/DBoW2/DBoW2/BowVector.h"
#include "thirdparty/DBoW2/DBoW2/FeatureVector.h"

class Mappoint;
class Camera;
class StereoCamera;

namespace ORB_SLAM2 { class ORBextractor; };

class MementoStereoFrame;

class Frame {
public:
  Frame(cv::Mat gray, int id, const Camera* camera, ORB_SLAM2::ORBextractor* extractor);

  virtual ~Frame();

  int GetIndex(const Mappoint* mp) const;
  int GetIndex() const;
  int GetKeyframeIndex() const;

  const std::vector<cv::KeyPoint>& GetKeypoints() const;
  const cv::KeyPoint& GetKeypoint(const Mappoint* mp) const;

  Mappoint* GetMappoint(int kpt_index) const;
  const std::set<Mappoint*> GetMappoints() const;
  const std::vector<Mappoint*>& GetVecMappoints() const;
  void SetMappoint(Mappoint* mp, int kpt_index);
  void SetMappoitIfEmpty(Mappoint* mp, int index);
  int EraseMappoint(Mappoint* mp);

  void SetTcw(const g2o::SE3Quat& Tcw);
  g2o::SE3Quat GetTcw() const;

  void SetBrightness(const Eigen::Vector2d& brightness);
  Eigen::Vector2d GetBrightness() const;

  const BrightenSE3 GetBrightenPose() const;

  void ComputeBoW(ORB_SLAM2::ORBVocabulary* voc);
  const DBoW2::BowVector& GetBowVec() const;
  const DBoW2::FeatureVector& GetFeatureVec() const;

  bool IsKeyframe() const;
  void SetKeyframe(ORB_SLAM2::ORBVocabulary* voc, int kf_id);
  bool IsInFrame(const Eigen::Vector2d& uv) const;
  void GetNeighbors(std::set<Frame*>& neighbor_keyframes,
                    std::set<Mappoint*>& covisible_mappoints,
                    int min_covisibility = -1
                    ) const;
  void GetNeighbors(std::set<Frame*>& neighbor_keyframes,
                    int min_covisibility = -1
                    ) const;
  void GetNeighbors(std::vector<Frame*>& neighbor_keyframes,
                    int min_covisibility = -1
                    ) const;
  std::vector<Frame*> GetBestCovisibilityKeyFrames(int num_of_keyframes = -1) const;
  void SetLoopQuery(Frame* loop_query);
  Frame* GetLoopQuery() const;
  void SetLoopWords(size_t loop_words);
  size_t GetLoopWords() const;
  void SetLoopScore(float score);
  float GetLoopScore() const;
  virtual void ReduceMemSize();

  const cv::Mat GetDescription(int i) const;

  std::set<int> SearchRadius(const Eigen::Vector2d& uv, double radius) const;
  std::vector<std::vector<int> > SearchRadius(const flann::Matrix<double>& points,
                                              double radius) const;
  void SearchNeareast(const Eigen::Vector2d& uv, int& kpt_index, double& distance) const;

  const Eigen::Vector2d& GetNormalizedPoint(size_t kpt_idx) const; // << normalized undistorted

  cv::Mat GetImage() const;

  const Camera* GetCamera() const;

  const ORB_SLAM2::ORBextractor* GetExtractor() const;

  virtual bool GetDepth(int kpt_index, Eigen::Vector3d& Xc) const = 0;

  static size_t frame_n_;
  static size_t keyframe_n_;

  cv::Mat PlotProjection(const std::set<Mappoint*>& mappoints) const;

protected:
  cv::Mat gray_;
  const int id_;
  int kf_id_;

  std::vector<cv::KeyPoint> keypoints_;
  EigenVector<Eigen::Vector2d> normalized_keypoints_;
  flann::Matrix<double> flann_keypoints_;
  flann::Index<flann::L2<double> >* flann_kdtree_;

  cv::Mat descriptions_; // (row, col) for (nfeatures, Dim)

  std::vector<Mappoint*> mappoints_;
  std::map<const Mappoint*, int> mappoints_index_;
  BrightenSE3* estimation_;
  DBoW2::BowVector bowvec_;
  DBoW2::FeatureVector featvec_;
  Frame* loop_query_;
  size_t loop_words_;
  float loop_score_;

  const Camera*const camera_;
  ORB_SLAM2::ORBextractor*const extractor_;

  mutable std::mutex mutex_;
};

class StereoFrame : public Frame {
public:

  StereoFrame(cv::Mat gray, cv::Mat gray_right, int id, const StereoCamera* camera, ORB_SLAM2::ORBextractor* extractor);

  StereoFrame(MementoStereoFrame* memento, const Camera* camera, ORB_SLAM2::ORBextractor* extractor, ORB_SLAM2::ORBVocabulary* voc);

  virtual ~StereoFrame();

  const cv::KeyPoint GetRightKeypoint(const Mappoint* mp) const;
  const std::vector<cv::KeyPoint>& GetRightKeypoints() const;
  const cv::Mat GetRightDescription(int i) const;
  const std::vector<int> GetStereoCorrespond() const;
  void SetStereoCorrespond(const std::vector<int> & correspond );

  void ExtractRightKeypoints();

  virtual bool GetDepth(int kpt_index, Eigen::Vector3d& Xc) const;
  cv::Mat PlotStereoMatch() const;

  virtual void ReduceMemSize();
private:
  cv::Mat gray_right_; // Reserve it until stereo match.
  std::vector<int> stereo_correspond_;
  std::vector<cv::KeyPoint> r_keypoints_;
  cv::Mat r_descriptions_;

};

std::set<Mappoint*> SupplyMappoints(Frame* frame);

/*

class RgbdFrame : public Frame {
public:
  virtual bool GetDepth(int kpt_index, Eigen::Vector3d& Xc) const;
private:
  cv::Mat depth_;
};


*/

#endif
