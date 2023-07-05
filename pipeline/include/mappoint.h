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

#ifndef MAPPOINT_H_
#define MAPPOINT_H_
#include "stdafx.h"

class MementoMappoint;
class Frame;

class Mappoint{
public:
  friend class MementoMappoint;

  Mappoint(Frame* ref, double invd, int id);

  Mappoint(Frame* ref, MementoMappoint* memento);

  void SetInvD(double invd);

  double GetDepth() const;

  virtual Eigen::Vector3d GetXw() const;

  cv::Mat GetDescription() const;
  cv::KeyPoint GetKeypoint() const;

  Frame* GetRefFrame() const;

  void AddKeyframe(Frame* keyframe); // This method must be called only by Frame::SetKeyframe or Frame::SetMappoint.
  const std::set<Frame*>& GetKeyframes() const;

  int GetIndex() const;

  void SetBad();
  bool IsBad() const;

public:
  static size_t n_;

protected:
  Frame*const ref_;
  std::set<Frame*> keyframes_;

  const int id_;
  double invd_;
  bool bad_flag_;
  mutable std::mutex mutex_;
};


#endif
