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

#ifndef LOOPCLOSER_H_
#define LOOPCLOSER_H_

#include "frame.h"
#include "stdafx.h"

class PipelineMap;
class BasicPoseTracker;
class IndirectPoseTracker;

class LoopCloser {
public:
  LoopCloser(PipelineMap* map,
             std::shared_ptr<IndirectPoseTracker> pose_tracker);

  bool GetRelativePose(const Frame* curr_frame,
                       const std::vector<Frame*>& loop_candidates,
                       Frame*& loop_frame,
                       g2o::SE3Quat& T_curr_loop) const;

  std::set<Frame*> CloseLoop(Frame* curr_frame,
                             Frame* loop_frame,
                             const g2o::SE3Quat& T_curr_loop);

private:

  void CombineNeighborMappoints(Frame* curr_frame, Frame* loop_frame) const;

  PipelineMap*const map_;
  EigenMap<std::pair<Frame*, Frame*>, g2o::SE3Quat> closed_loops_;

  std::shared_ptr<IndirectPoseTracker> pose_tracker_;
};

#endif
