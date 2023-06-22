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

#include "../include/seg.h"
#include "../include/util.h"
#include "../include/seg.h"

#include "dataset.h"
#include "orb_extractor.h"
#include "frame.h"
#include "camera.h"
//#include "pipeline.h"
//#include "qmap_viewer.h"
//#include "common.h"
//#include <QApplication>
//#include <QWidget>

cv::Mat Score2Binary(cv::Mat flow_difference) {
  const int rows = flow_difference.rows;
  const int cols = flow_difference.cols;
  cv::Mat binary_edges = cv::Mat::zeros(rows, cols, CV_8UC1); 


  return binary_edges;
}

int main(int argc, char** argv){
  //Seq :"02",("13" "20");
  std::string fn_config = GetPackageDir()+"/config/kitti.yaml";
  cv::FileStorage config(fn_config, cv::FileStorage::READ);
  std::string seq(argv[1]);

  KittiDataset dataset(seq);
  const auto& Tcws = dataset.GetTcws();
  if(Tcws.empty()){
    std::cout << "Seq" << seq << " with no ground truth trajectory." << std::endl;
  }
  const auto& D = dataset.GetCamera()->GetD();
  std::cout << "Distortion = " << D.transpose() << std::endl;
  const StereoCamera* camera = dynamic_cast<const StereoCamera*>(dataset.GetCamera());
  assert(camera);
  Seg seg;

  bool stop = true;
  g2o::SE3Quat Tc0w;
  for(int i=0; i<dataset.Size(); i+=1){
    const cv::Mat gray   = dataset.GetImage(i);
    const cv::Mat gray_r = dataset.GetRightImage(i);
    g2o::SE3Quat Tcw;
    if(!Tcws.empty())
      Tcw = Tcws.at(i);
    else{
      g2o::SE3Quat Tc1c0;
      Tc1c0.setTranslation(Eigen::Vector3d(0.,0.,-.2) );
      Tcw = Tc1c0 * Tc0w;
    }
    Tc0w = Tcw;
    seg.Put(gray, gray_r, Tcw, *camera );
    char c = cv::waitKey(stop?0:1);
    if(c == 'q')
      break;
    else if (c == 's')
      stop = !stop;
  }

  return 1;
}
