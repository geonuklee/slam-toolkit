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

#ifndef PIPELINE_H_
#define PIPELINE_H_

#include "ORBVocabulary.h"
#include "stdafx.h"
#include <thread>

class MementoPipeline;
class LoopCloser;
class LoopDetector;
class Frame;
class PipelineMap;
class Camera;

namespace ORB_SLAM2{ class ORBextractor; }

ORB_SLAM2::ORBextractor* CreateExtractor();
ORB_SLAM2::ORBVocabulary* CreateVocabulary();

struct FrameInfo{
  double elapsed_ms_;
};

class PipelineViewer {
public:
  virtual void OnSetKeyframe(Frame* kf) = 0;
  virtual void UpdateMappoints(Frame* frame){}
  virtual void OnFrame(Frame* frame, const FrameInfo& info) = 0;
};

class AbstractPipeline{
public:
  virtual g2o::SE3Quat Track(cv::Mat im_left, cv::Mat im_right) = 0;
  virtual PipelineMap* GetMap() const = 0;
  virtual void Save() const = 0;

  virtual void AddViewer(PipelineViewer* viewer) = 0;
};

class BasicPoseTracker;
class BasicLocalMapper;

class Pipeline : public AbstractPipeline{
public:
  Pipeline(const Camera* camera);

  virtual ~Pipeline();
  g2o::SE3Quat Track(cv::Mat im_left, cv::Mat im_right);

  PipelineMap* GetMap() const { return map_; }


  virtual void AddViewer(PipelineViewer* viewer);
  virtual void Save() const;

private:
  void mapping();

  bool IsKeyframe(Frame* frame) const;

  void AddMappoints(Frame* frame,
                    const std::set<Frame*>& neighbor_keyframes) const;

  const Camera*const camera_;
  ORB_SLAM2::ORBextractor*const extractor_;

  BasicPoseTracker* pose_tracker_;
  BasicLocalMapper* local_mapper_;
  PipelineMap*const map_;

  LoopDetector* loop_detetor_;
  LoopCloser* loop_closer_;

  Frame* latest_keyframe_;
  std::vector<PipelineViewer*> viewers_;

  std::thread* mapping_thread_;
  std::queue<Frame*> new_keyframes_;
  bool exit_flag_;
  mutable std::mutex mutex_new_keyframes_;
};

bool DoFrameNeedsNewMappoints(const Frame* frame);

#endif
