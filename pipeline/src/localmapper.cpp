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

#include "localmapper.h"

#include "method.h"
#include "pipeline_map.h"
#include "orb_extractor.h"
#include "mappoint.h"
#include "frame.h"
#include "camera.h"

BasicLocalMapper::BasicLocalMapper(const std::shared_ptr<StandardMethod> method)
 : method_(method)
{

}

void BasicLocalMapper::InitializeGraph(const PipelineMap* map, Frame* curr_frame,
                    g2o::SparseOptimizer& optimizer,
                    std::map<Frame*, g2o::OptimizableGraph::Vertex*>& v_poses,
                    std::map<Mappoint*, g2o::OptimizableGraph::Vertex*>& v_mappoints,
                    std::map<g2o::OptimizableGraph::Edge*, std::pair<Frame*,Mappoint*> >& measurment_edges,
                    std::map<g2o::OptimizableGraph::Edge*, Mappoint*>& anchor_measurment_edges
                   ) {

  //>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
  map->Lock();
  std::set<Frame*> local_frames;
  std::map<Frame*, int> frame_indecies;
  std::set<Mappoint*> local_mappoints;
  std::set<Mappoint*> curr_visible_only;
  {
    curr_frame->GetNeighbors(local_frames,  curr_visible_only);
  }
  {
    auto latests = map->GetLatestFrames(5, true);
    for(Frame* frame : latests)
      local_frames.insert(frame);
  }
  local_frames.insert(curr_frame);
  Frame* oldest = curr_frame;
  for(Frame* frame : local_frames){
    auto vertex = method_->CreatePoseVertex(optimizer, frame);
    v_poses[frame] = vertex;
    frame_indecies[frame] = frame->GetIndex();
    if(frame->GetIndex() < oldest->GetIndex())
      oldest = frame;
    std::set<Mappoint*> mps = frame->GetMappoints();
    for(Mappoint* mp : mps)
      local_mappoints.insert(mp);
  }

  if(!v_poses.empty())
    v_poses.at(oldest)->setFixed(true);

  for(Mappoint* mp : local_mappoints){
    const std::set<Frame*>& keyframes_of_mp = mp->GetKeyframes();
    if(keyframes_of_mp.size() < 2)
      continue;
    auto v_mp = method_->CreateStructureVertex(optimizer, mp);
    v_mp->setMarginalized(true);
    v_mp->setFixed(!curr_visible_only.count(mp));
    v_mappoints[mp] = v_mp;

    for(Frame* kf : keyframes_of_mp){
      g2o::OptimizableGraph::Vertex* v_pose = nullptr;
      if( !v_poses.count(kf) ){
        v_pose = method_->CreatePoseVertex(optimizer, kf);
        v_pose->setFixed(true);
        v_poses[kf] = v_pose;
      }
      else
        v_pose = v_poses.at(kf);

      g2o::OptimizableGraph::Edge* edge = method_->CreateMeasurementEdge(kf, mp, true);
      edge->setVertex(0, v_mp);
      edge->setVertex(1, v_pose);
      optimizer.addEdge(edge);
      measurment_edges[edge] = std::make_pair(kf, mp);
    }

    {
      Frame* rkf = mp->GetRefFrame();
      g2o::OptimizableGraph::Vertex* v_ref_pose = nullptr;
      if( !v_poses.count(rkf) ){
        v_ref_pose = method_->CreatePoseVertex(optimizer, rkf);
        v_ref_pose->setFixed(true);
        v_poses[rkf] = v_ref_pose;
      }
      else
        v_ref_pose = v_poses.at(rkf);
      g2o::OptimizableGraph::Edge* anchor_edge = method_->CreateAnchorMeasurementEdge(rkf, mp);
      anchor_edge->setVertex(0, v_mp);
      anchor_edge->setVertex(1, v_ref_pose);
      anchor_measurment_edges[anchor_edge] = mp;
    }
  }
  map->UnLock();
}

void BasicLocalMapper::Optimize(const PipelineMap* map, Frame* curr_frame, int n_iter) {

   g2o::SparseOptimizer optimizer;
   method_->SetSolver(StandardMethod::LM, optimizer);

   std::map<Frame*, g2o::OptimizableGraph::Vertex*> v_poses;
   std::map<Mappoint*, g2o::OptimizableGraph::Vertex*> v_mappoints;
   std::map<g2o::OptimizableGraph::Edge*, std::pair<Frame*,Mappoint*> > measurment_edges;
   std::map<g2o::OptimizableGraph::Edge*, Mappoint*> anchor_measurment_edges;

   InitializeGraph(map, curr_frame, optimizer, v_poses, v_mappoints, measurment_edges, anchor_measurment_edges);
   if(v_poses.size() < 3)
     return;

   optimizer.initializeOptimization();
   optimizer.optimize(n_iter);

   // Retrieve
   //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
   map->Lock();
   for(auto it_poses : v_poses){
     g2o::OptimizableGraph::Vertex* vertex = it_poses.second;
     if(vertex->fixed())
       continue;
     Frame* frame = it_poses.first;
     const bool already_locked = true;
     if(map->HasFrame(frame, already_locked) ) // if not expired yet
       method_->RetrivePose(vertex, frame);
   }

   for(auto it_mp : v_mappoints){
     g2o::OptimizableGraph::Vertex* vertex = it_mp.second;
     if(vertex->fixed())
       continue;
     Mappoint* mp = it_mp.first;
     method_->RetriveStructure(vertex, mp);
   }
   map->UnLock();
   //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
   return;
}

