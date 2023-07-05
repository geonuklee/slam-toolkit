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

#ifndef LOCALMAPPER_H_
#define LOCALMAPPER_H_

class StandardMethod;
class PipelineMap;
class Frame;
class Mappoint;

#include <g2o/core/sparse_optimizer.h>

class StandardLocalMapper{
public:
  StandardLocalMapper(const std::shared_ptr<StandardMethod> method);

  void InitializeGraph(const PipelineMap* map, Frame* curr_frame,
                           g2o::SparseOptimizer& optimizer,
                           std::map<Frame*, g2o::OptimizableGraph::Vertex*>& v_poses,
                           std::map<Mappoint*, g2o::OptimizableGraph::Vertex*>& v_mappoints,
                           std::map<g2o::OptimizableGraph::Edge*, std::pair<Frame*,Mappoint*> >& measurment_edges,
                           std::map<g2o::OptimizableGraph::Edge*, Mappoint*>& anchor_measurment_edges
                          );

  virtual void Optimize(const PipelineMap* map, Frame* curr_frame, int n_iter);

protected:
  const std::shared_ptr<StandardMethod> method_;
};

#endif
