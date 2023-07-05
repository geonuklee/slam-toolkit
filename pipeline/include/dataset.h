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

#ifndef DATASET_H_
#define DATASET_H_
#include "stdafx.h"
#include "common.h"
#include "camera.h"

namespace ORB_SLAM2{
class ORBextractor;
}

class Dataset {
public:
  virtual const EigenMap<int, g2o::SE3Quat>& GetTcws() const = 0;
  virtual cv::Mat GetImage(int i) const = 0;
  virtual int Size() const = 0;
  virtual const Camera* GetCamera() const = 0;

  //virtual ORB_SLAM2::ORBextractor* GetExtractor() const = 0; // Suggest proper extractor for given dataset
};

class StereoDataset : public Dataset{
public:
  virtual cv::Mat GetRightImage(int i) const = 0;

protected:
};

class KittiDataset : public StereoDataset {
public:
  KittiDataset(std::string seq);
  virtual ~KittiDataset();

  virtual cv::Mat GetImage(int i) const;
  virtual cv::Mat GetRightImage(int i) const; 
  virtual int Size() const;
  virtual const EigenMap<int, g2o::SE3Quat>& GetTcws() const;
  virtual const Camera* GetCamera() const;

  // virtual ORB_SLAM2::ORBextractor* GetExtractor() const;
  //ORB_SLAM2::ORBextractor* extractor_;
private:
  std::map<int, std::string> im0_filenames_;
  std::map<int, std::string> im1_filenames_;

  EigenMap<int, g2o::SE3Quat> Tcws_;

  StereoCamera* camera_;
};

#endif
