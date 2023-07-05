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

#include "stdafx.h"
#include "dataset.h"
//#include "orb_extractor.h"
#include "frame.h"
#include "camera.h"
#include "pipeline.h"

#include "qmap_viewer.h"
#include "common.h"

#include <QApplication>
#include <QWidget>


int main(int argc, char** argv){
  std::string fn_config = GetPackageDir()+"/config/kitti.yaml";
  cv::FileStorage config(fn_config, cv::FileStorage::READ);
  std::string seq = GetFromFileStorage<std::string>(config, "seq");
  KittiDataset dataset(seq);
  Pipeline pipeline(dataset.GetCamera());
  QApplication app(argc,argv);
  QPipelinePlayer pipeline_player(&pipeline, &dataset);
  QMapViewer widget(&pipeline_player, config);
  CvViewer viewer(&pipeline);
  widget.SetDataset(&dataset);
  pipeline.AddViewer(&widget);
  widget.show();
  pipeline_player.Start();
  app.exec();
  return 1;
}
