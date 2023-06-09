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

#ifndef QMAP_VIEWER_
#define QMAP_VIEWER_

#include <QWidget>
#include <QTimer>
#include <QVTKWidget.h>
#include <vtkSmartPointer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkActor.h>
#include <vtkCornerAnnotation.h>

#include "stdafx.h"
#include "pipeline.h"
#include "common.h"

class Dataset;

class TrjDrawer {
public:
  TrjDrawer();
  void Draw(EigenMap<int, g2o::SE3Quat>& Tcws);
  vtkSmartPointer<vtkActor> GetActor() const;

private:
  vtkSmartPointer<vtkActor> actor_;
};

class vtkPoints;
class vtkVertexGlyphFilter;
class Mappoint;

class QPipelinePlayer : public QWidget {
public:
  QPipelinePlayer(AbstractPipeline* pipeline, Dataset* dataset);
  AbstractPipeline* GetPipeline() const { return pipeline_; }
  bool IsActive() const;

  void Start();

  void Stop();

private:
  void OnTimer();
  AbstractPipeline*const pipeline_;
  Dataset*const dataset_;
  QTimer*const timer_;
};

class QMapViewer : public QWidget, public PipelineViewer {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  QMapViewer(QPipelinePlayer* pipeline_player,
             const cv::FileStorage& config);
  ~QMapViewer();

  void SetDataset(Dataset* dataset);

  void UpdateMappoints(Frame* frame);

private:
  virtual void keyPressEvent(QKeyEvent *event);
  virtual bool eventFilter(QObject *watched, QEvent *event);
  void Save();

  virtual void OnSetKeyframe(Frame* kf);
  virtual void OnFrame(Frame* frame, const FrameInfo& info);


  void ResetCamera();

  QVTKWidget*const qvtk_widget_;
  vtkSmartPointer<vtkRenderWindow> render_window_;
  vtkSmartPointer<vtkRenderer> renderer_;

  Dataset* dataset_;
  const AbstractPipeline*const pipeline_;

  TrjDrawer gt_drawer_;
  TrjDrawer est_drawer_;
  g2o::SE3Quat Tgt_est_;

  vtkSmartPointer<vtkActor> coord_actor_;
  vtkSmartPointer<vtkCornerAnnotation> corner_annotation_;

  QPipelinePlayer*const pipeline_player_;
  const cv::FileStorage config_;

  vtkSmartPointer<vtkActor> mappoints_actor_;
  vtkSmartPointer<vtkPoints> points_;
  std::map<Mappoint*, int> drawed_mappoints_;
  vtkSmartPointer<vtkVertexGlyphFilter> vertexglyphfilter_;
};

class CvViewer : public PipelineViewer {
public:
  CvViewer(Pipeline* pipeline);

  virtual void OnSetKeyframe(Frame* kf){ };
  virtual void OnFrame(Frame* frame, const FrameInfo& info);

private:
  const AbstractPipeline*const pipeline_;
};


#endif
