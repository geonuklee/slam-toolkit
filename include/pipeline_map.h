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

#ifndef PIPELINE_MAP_
#define PIPELINE_MAP_

#include "ORBVocabulary.h"

#include "stdafx.h"

class Frame;
class MementoMap;
class Camera;
namespace ORB_SLAM2 {
class ORBextractor;
}


class PipelineMap {
public:
#ifdef MEMENTO_H_
  friend class MementoMap;
  PipelineMap(ORB_SLAM2::ORBVocabulary* voc,
              const Camera* camera,
              ORB_SLAM2::ORBextractor* extractor,
              const MementoMap* memento);
#endif

  PipelineMap(ORB_SLAM2::ORBVocabulary* voc);
  ~PipelineMap();

  const std::map<int, Frame*> GetFrames(bool already_locked = false) const;
  size_t GetFramesNumber(bool already_locked = false) const;

  bool HasFrame(Frame* frame, bool already_locked = false) const;
  Frame* GetFrame(int frame_idx, bool already_locked = false) const;

  Frame* GetLatestFrame() const;
  std::set<Frame*> GetLatestFrames(int number, bool already_locked=false) const;

  void CullingOldFrames(int reserve_range);

  void AddFrame(Frame* frame, bool already_locked = false);

  std::vector<Frame*> DetectLoopCandidates(Frame* pKF, float minScore, bool already_locked=false) const;

  ORB_SLAM2::ORBVocabulary* GetVocaBulary() const; 

  void Lock() const { mutex_.lock(); }
  void UnLock() const { mutex_.unlock(); }

private:
  std::map<int, Frame*> frames_;
  std::set<Frame*> set_frames_;

  std::vector<std::list<Frame*> > inverted_file_;

  ORB_SLAM2::ORBVocabulary*const voc_;
  mutable std::mutex mutex_;
};

#endif
