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

#ifndef METHOD_H_ 
#define METHOD_H_

#include <g2o/core/sparse_optimizer.h>

class PipelineMap;
class Frame;
class Mappoint;

class StandardMethod {
public:
  enum OptimizationMethod{ GN,  LM };

  virtual ~StandardMethod(){}
  virtual void SetSolver(OptimizationMethod optimization_method, g2o::SparseOptimizer& optimizer) = 0;
  virtual g2o::OptimizableGraph::Edge* CreateAnchorMeasurementEdge(const Frame* ref_frame, Mappoint* mappoint) = 0;
  virtual g2o::OptimizableGraph::Edge* CreateMeasurementEdge(const Frame* frame, Mappoint* mappoint, bool set_measurement_from_frame) = 0;

  virtual g2o::OptimizableGraph::Vertex* CreatePoseVertex(g2o::SparseOptimizer& optimizer, Frame* frame) = 0;
  virtual g2o::OptimizableGraph::Vertex* CreateStructureVertex(g2o::SparseOptimizer& optimizer, Mappoint* mappoint) = 0;

  virtual void RetrivePose(const g2o::OptimizableGraph::Vertex* vertex, Frame* frame) = 0;
  virtual void RetrivePose(const g2o::OptimizableGraph::Vertex* vertex, void* estimation) = 0;
  virtual void SetPose(const void* predict, g2o::OptimizableGraph::Vertex* vertex) = 0;

  virtual void RetriveStructure(const g2o::OptimizableGraph::Vertex* vertex, Mappoint* mappoint) = 0;
};

class StereoFrame;
class IndirectStereoMethod : public StandardMethod {
public:
  IndirectStereoMethod(const std::vector<float>& inv_scales_sigma2);
  const std::vector<float>& GetInvScaleSigma2() const {return inv_scales_sigma2_; }

protected:
  virtual void SetSolver(OptimizationMethod optimization_method, g2o::SparseOptimizer& optimizer);
  virtual g2o::OptimizableGraph::Edge* CreateAnchorMeasurementEdge(const Frame* ref_frame, Mappoint* mappoint);
  virtual g2o::OptimizableGraph::Edge* CreateMeasurementEdge(const Frame* frame, Mappoint* mappoint, bool set_measurement_from_frame);

  virtual g2o::OptimizableGraph::Vertex* CreatePoseVertex(g2o::SparseOptimizer& optimizer, Frame* frame);
  virtual g2o::OptimizableGraph::Vertex* CreateStructureVertex(g2o::SparseOptimizer& optimizer, Mappoint* mappoint);
  virtual void RetrivePose(const g2o::OptimizableGraph::Vertex* vertex, Frame* frame);
  virtual void RetrivePose(const g2o::OptimizableGraph::Vertex* vertex, void* estimation);
  virtual void SetPose(const void* predict, g2o::OptimizableGraph::Vertex* vertex);
  virtual void RetriveStructure(const g2o::OptimizableGraph::Vertex* vertex, Mappoint* mappoint);

  const std::vector<float> inv_scales_sigma2_;
};

class DirectPyramid;

class DirectStereoMethod : public StandardMethod {
public:
  DirectStereoMethod();

  std::shared_ptr<DirectPyramid> GeyPyramid( )const;

protected:
  virtual void SetSolver(OptimizationMethod optimization_method, g2o::SparseOptimizer& optimizer);
  virtual g2o::OptimizableGraph::Edge* CreateAnchorMeasurementEdge(const Frame* ref_frame, Mappoint* mappoint);
  virtual g2o::OptimizableGraph::Edge* CreateMeasurementEdge(const Frame* frame, Mappoint* mappoint, bool set_measurement_from_frame);

  virtual g2o::OptimizableGraph::Vertex* CreatePoseVertex(g2o::SparseOptimizer& optimizer, Frame* frame);
  virtual g2o::OptimizableGraph::Vertex* CreateStructureVertex(g2o::SparseOptimizer& optimizer, Mappoint* mappoint);
  virtual void RetrivePose(const g2o::OptimizableGraph::Vertex* vertex, Frame* frame);
  virtual void RetrivePose(const g2o::OptimizableGraph::Vertex* vertex, void* estimation);
  virtual void SetPose(const void* predict, g2o::OptimizableGraph::Vertex* vertex);
  virtual void RetriveStructure(const g2o::OptimizableGraph::Vertex* vertex, Mappoint* mappoint);

  std::shared_ptr<DirectPyramid> pyramid_;
  const double huber_delta_;
};




#endif

