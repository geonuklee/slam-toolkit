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

int TestKitti(int argc, char** argv) {
  //Seq :"02",("13" "20");
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
  for(int i=0; i<dataset.Size(); i+=1){
    std::cout << "F# " << i << std::endl;
    const cv::Mat rgb   = dataset.GetImage(i, cv::IMREAD_COLOR);
    const cv::Mat rgb_r = dataset.GetRightImage(i, cv::IMREAD_COLOR);
    seg.Put(rgb, rgb_r, *camera );
    char c = cv::waitKey(stop?0:1);
    if(c == 'q')
      break;
    else if (c == 's')
      stop = !stop;
  }
  std::cout << "Done. The end of the sequence" << std::endl;
  return 1;
}

int TestWaymodataset(int argc, char** argv) {
  // 주어진 RGB+Dense Depthmap에서 
  std::string dataset_path = GetPackageDir()+ "/../thirdparty/waymo-dataset/output/";
  std::string seq = "segment-10247954040621004675_2180_000_2200_000_with_camera_labels"; // TODO Batch실행?
  KittiDataset dataset(seq,dataset_path);
  const DepthCamera* camera = dynamic_cast<const DepthCamera*>(dataset.GetCamera());
  assert(camera);
  Seg seg;
  //std::cout << "Intrinsic = \n" << camera->GetK() << std::endl;
  const EigenMap<int, g2o::SE3Quat>& Tcws = dataset.GetTcws();
  bool stop = false;
  for(int i=0; i<dataset.Size(); i+=1){
    const auto Twc = Tcws.at(i).inverse();
    //std::cout << "t=" << Twc.translation().transpose() << std::endl;
    const cv::Mat rgb   = dataset.GetImage(i,cv::IMREAD_UNCHANGED);
    const cv::Mat depth = dataset.GetDepthImage(i);
    seg.Put(rgb, depth, *camera );

    char c = cv::waitKey(stop?0:100);
    if(c == 'q')
      break;
    else if (c == 's')
      stop = !stop;
  }
  return 1;
}

int main(int argc, char** argv){
  return TestKitti(argc, argv); // TODO depth update가 제대로 안된다
  //return TestWaymodataset(argc, argv);

  return 1;
}
