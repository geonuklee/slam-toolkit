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

#ifndef POSETRACKER_H_
#define POSETRACKER_H_

#include "stdafx.h"
#include "common.h"

#include <g2o/core/sparse_optimizer.h>

class Mappoint;
class Frame;

// Posetracker, Loopcloser 의 filter를 아래 형태로 통일.
class OutlierFilter {
public:
  virtual std::list<std::pair<size_t, Mappoint*> > GetOutlier(const Frame* frame,
                                                              const void* expected_pose
                                                              ) = 0;
};

class ReprojectionFilter : public OutlierFilter {
public:
  ReprojectionFilter(double max_reprojection_error);
  virtual std::list<std::pair<size_t, Mappoint*> > GetOutlier(const Frame* frame,
                                                              const void* expected_pose
                                                             );
private:
  const double max_reprojection_error_;
};

class PhotometricErrorFilter : public OutlierFilter {
public:
  PhotometricErrorFilter(double max_error);
  virtual std::list<std::pair<size_t, Mappoint*> > GetOutlier(const Frame* frame,
                                                              const void* expected_pose
                                                             );
private:
  const double max_error_;
};

// TODO pose thread와 local mapper thread가 동시에 지우면 문제됨.
// 지금은 잘 관리해서 쓰지만, 자동적인 안전장치가 필요.
// 임의의 pose type, frame type에 대처하는 posetracker의 빌드시간 절약을 위해 template 대신 void* 를 사용.
class StandardMethod;
class StandardPoseTracker {
public:
  StandardPoseTracker(StandardMethod* method);
  virtual ~StandardPoseTracker() {}

  // Update pose
  virtual void Track(const std::set<Mappoint*>& mappoints,
                     const void* predict,
                     void* estimation,
                     int n_iter,
                     Frame* frame
                     );

  virtual void BeforeEstimation(const std::set<Mappoint*>& mappoints,
                                const void* predict,
                                const Frame* frame) = 0;

  // Estiation only
  virtual void EstimatePose(const std::set<Mappoint*>& mappoints,
                            const void* predict,
                            void* estimation,
                            int n_iter,
                            const Frame* frame
                            );

protected:
  virtual bool HasMeasurement(const Frame* frame, Mappoint* mp) = 0;
  virtual void SetMeasurement(const Frame* frame, Mappoint* mp, g2o::OptimizableGraph::Edge* edge) = 0;

  void InitializeGraph(const std::set<Mappoint*>& mappoints,
                       const void* predict,
                       const Frame* frame,
                       g2o::SparseOptimizer& optimizer,
                       g2o::OptimizableGraph::Vertex*& v_pose,
                       std::map<g2o::OptimizableGraph::Edge*, std::pair<const Frame*,Mappoint*> >& measurment_edges
                      );

  virtual void RetriveEstimation(const void* estimation, Frame* frame) = 0;

  const std::shared_ptr<StandardMethod> method_;
};


class IndirectPoseTracker : public StandardPoseTracker {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  IndirectPoseTracker(const std::vector<float>& inv_scales_sigma2);
  const std::map<Mappoint*, int>& GetMatches() const { return matched_mappoints_; }

  virtual void BeforeEstimation(const std::set<Mappoint*>& mappoints,
                                const void* predict,
                                const Frame* frame);

protected:
  virtual bool HasMeasurement(const Frame* frame, Mappoint* mp);
  virtual void SetMeasurement(const Frame* frame, Mappoint* mp, g2o::OptimizableGraph::Edge* edge);

  virtual void RetriveEstimation(const void* estimation, Frame* frame);

  g2o::SE3Quat predicted_Tcw_;
  std::map<Mappoint*, int> matched_mappoints_;
  std::vector<cv::KeyPoint> keypoints_;
};


class BrightenDirectPoseTracker : public StandardPoseTracker {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  BrightenDirectPoseTracker(double min_search_radius);

  virtual void BeforeEstimation(const std::set<Mappoint*>& mappoints,
                                const void* predict,
                                const Frame* frame);

  virtual void EstimatePose(const std::set<Mappoint*>& mappoints,
                            const void* predict,
                            void* estimation,
                            int n_iter,
                            const Frame* frame
                            );

protected:
  virtual bool HasMeasurement(const Frame* frame, Mappoint* mp);
  virtual void SetMeasurement(const Frame* frame, Mappoint* mp, g2o::OptimizableGraph::Edge* edge);

  virtual void RetriveEstimation(const void* estimation, Frame* frame);

  double min_search_radius_;
  BrightenSE3 predicted_pose_;
  std::set<Mappoint*> mappoints_;
};




#endif
