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

#include "qmap_viewer.h"
#include "dataset.h"
#include "optimizer.h"
#include "frame.h"
#include "mappoint.h"
#include "pipeline_map.h"

#include <QVBoxLayout>
#include <QToolBar>
#include <QEvent>
#include <QKeyEvent>

#include <vtkPolyData.h>
#include <vtkCellArray.h>
#include <vtkCellData.h>
#include <vtkLine.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkCamera.h>
#include <vtkTextProperty.h>
#include <vtkVertexGlyphFilter.h>
#include <QHBoxLayout>
#include <QPushButton>

void SetActorTf(vtkSmartPointer<vtkActor> actor, const g2o::SE3Quat& Twc){
	Eigen::AngleAxisd rot(Twc.rotation());
	Eigen::Vector3d axis = rot.axis();
	double angle = rot.angle();
	Eigen::Vector3d t = Twc.translation();
	actor->SetOrientation(0.0, 0.0, 0.0);
	actor->RotateWXYZ(angle*180.0 / M_PI, axis[0], axis[1], axis[2]);
	actor->SetPosition(t[0], t[1], t[2]);
	return;
}

vtkSmartPointer<vtkPolyData> CreateCoordPolyData(double length){
	vtkSmartPointer<vtkPolyData> poly_data = vtkSmartPointer<vtkPolyData>::New();
	poly_data->SetPoints(vtkSmartPointer<vtkPoints>::New());
	poly_data->SetLines(vtkSmartPointer<vtkCellArray>::New());
	vtkSmartPointer<vtkUnsignedCharArray> colors = vtkSmartPointer<vtkUnsignedCharArray>::New();
	colors->SetName("Colors");
	colors->SetNumberOfComponents(3);
	poly_data->GetPoints()->InsertNextPoint(0, 0, 0);
	for(int i = 0; i < 3; i++){
		unsigned char c[3] = { 0, };
		c[i] = 255;
		colors->InsertNextTupleValue(c);
		double p[3] = { 0.0, };
		p[i] = length;
		poly_data->GetPoints()->InsertNextPoint(p);
		vtkSmartPointer<vtkLine> line = vtkSmartPointer<vtkLine>::New();
		line->GetPointIds()->SetNumberOfIds(2);
		line->GetPointIds()->SetId(0, 0);
		line->GetPointIds()->SetId(1, i + 1);
		poly_data->GetLines()->InsertNextCell(line);
	}
	poly_data->GetCellData()->SetScalars(colors);
	return poly_data;
}

vtkSmartPointer<vtkActor> CreateCoordActor(double length, double width){
	vtkSmartPointer<vtkPolyData> poly_data = CreateCoordPolyData(length);
	vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
	mapper->SetInputData(poly_data);
	vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
	actor->SetMapper(mapper);
	actor->GetProperty()->SetLineWidth(width);
	return actor;
}


QPipelinePlayer::QPipelinePlayer(AbstractPipeline* pipeline, Dataset* dataset)
  :pipeline_(pipeline),
  dataset_(dataset),
  timer_(new QTimer(this))
{
  QHBoxLayout* hb = new QHBoxLayout;
  setLayout(hb);

  QPushButton* bt_play = new QPushButton("Play");
  QPushButton* bt_pause = new QPushButton("Pause");
  hb->addWidget(bt_play);
  hb->addWidget(bt_pause);

  QObject::connect(bt_play, &QPushButton::clicked, this, [=](){ Start(); });
  QObject::connect(bt_pause, &QPushButton::clicked, this, [=](){ Stop(); });
  QObject::connect(timer_, &QTimer::timeout, this, [=](){ OnTimer(); });
  timer_->setInterval(1);
}

bool QPipelinePlayer::IsActive() const {
  return timer_->isActive();
}

void QPipelinePlayer::Start() {
  timer_->start();
}

void QPipelinePlayer::Stop() {
  timer_->stop();
}

void QPipelinePlayer::OnTimer(){
  int n = 0;
  {
    PipelineMap* map = pipeline_->GetMap();
    if(map->GetFramesNumber() > 0)
      n = map->GetLatestFrame()->GetIndex()+1;
  }

  if(n >= dataset_->Size())
    return;

  // TODO 이걸 어떻게 일반화된 Player로 변경하지?
  // 일반화된 Player + Pipeline. 단 Pipeline을 종류별로 만드는것보단, 지금 구현에 dependency injection이 더 안전해보임.
  StereoDataset* stereo_dataset = dynamic_cast<StereoDataset*>(dataset_);
  cv::Mat im_left = stereo_dataset->GetImage(n);
  cv::Mat im_right = stereo_dataset->GetRightImage(n);
  pipeline_->Track(im_left, im_right);
  return;
}

QMapViewer::QMapViewer(QPipelinePlayer* pipeline_player,
                       const cv::FileStorage& config)
: qvtk_widget_(new QVTKWidget()),
  dataset_(nullptr),
  pipeline_(pipeline_player->GetPipeline()),
  coord_actor_(CreateCoordActor(40., 2)),
  corner_annotation_(vtkSmartPointer<vtkCornerAnnotation>::New()),
  pipeline_player_(pipeline_player),
  config_(config)
{
  auto vb = new QVBoxLayout(this);
  setLayout(vb);
  vb->addWidget(qvtk_widget_);

  auto toolbar = new QToolBar(this);
  vb->setMenuBar(toolbar);

  renderer_ = vtkSmartPointer<vtkRenderer>::New();
  render_window_ = vtkSmartPointer<vtkRenderWindow>::New();
  render_window_->AddRenderer(renderer_);
  qvtk_widget_->SetRenderWindow(render_window_);
  renderer_->SetBackground(1., 1., 1.);
  renderer_->AddActor(coord_actor_);

  corner_annotation_->SetLinearFontScaleFactor( 2 );
  corner_annotation_->SetNonlinearFontScaleFactor( 1 );
  corner_annotation_->SetMaximumFontSize( 20 );
  corner_annotation_->GetTextProperty()->SetColor( 0, 0, 1. );
  renderer_->AddViewProp( corner_annotation_ );

  resize(640,480);
  qvtk_widget_->installEventFilter(this);

  {
    mappoints_actor_ = vtkSmartPointer<vtkActor>::New();
    mappoints_actor_->GetProperty()->SetColor(0.8,0.8,0.8);
    mappoints_actor_->GetProperty()->SetPointSize(1.);
    renderer_->AddActor(mappoints_actor_);
    points_ = vtkSmartPointer<vtkPoints>::New();

    auto polydata = vtkSmartPointer<vtkPolyData>::New();
    polydata->SetPoints(points_);

    vertexglyphfilter_ = vtkSmartPointer<vtkVertexGlyphFilter>::New();
    vertexglyphfilter_->AddInputData(polydata);
    vertexglyphfilter_->Update();

    // Create a mapper and actor
    auto mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInputConnection(vertexglyphfilter_->GetOutputPort());

    mappoints_actor_->SetMapper(mapper);
  }
}

QMapViewer::~QMapViewer() {
  delete qvtk_widget_;
}

TrjDrawer::TrjDrawer() {
  actor_ = vtkSmartPointer<vtkActor>::New();
}

void TrjDrawer::Draw(EigenMap<int, g2o::SE3Quat>& Tcws) {
  vtkSmartPointer<vtkPolyData> poly_data = vtkSmartPointer<vtkPolyData>::New();
  poly_data->SetPoints(vtkSmartPointer<vtkPoints>::New());
  poly_data->SetLines(vtkSmartPointer<vtkCellArray>::New());

  for(auto it : Tcws){
    g2o::SE3Quat Twc = it.second.inverse();
    Eigen::Vector3d t = Twc.translation();
    poly_data->GetPoints()->InsertNextPoint(t.x(), t.y(), t.z());
    int n = poly_data->GetNumberOfPoints();
    if(n == 1)
      continue;
    vtkSmartPointer<vtkLine> line = vtkSmartPointer<vtkLine>::New();
    line->GetPointIds()->SetNumberOfIds(2);
    line->GetPointIds()->SetId(0, n-2);
    line->GetPointIds()->SetId(1, n-1);
    poly_data->GetLines()->InsertNextCell(line);
  }

  vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
  mapper->SetInputData(poly_data);
  actor_->SetMapper(mapper);
}

vtkSmartPointer<vtkActor> TrjDrawer::GetActor() const {
  return actor_;
}

void QMapViewer::SetDataset(Dataset* dataset) {
  dataset_ = dataset;
  EigenMap<int, g2o::SE3Quat> Tcws = dataset_->GetTcws();
  gt_drawer_.Draw(Tcws);

  auto actor = gt_drawer_.GetActor();
	actor->GetProperty()->SetLineWidth(2);
	actor->GetProperty()->SetColor(0.7, 0.7, 0.7);

  renderer_->AddActor(actor);
  ResetCamera();
  return;
}

void QMapViewer::ResetCamera() {
  vtkCamera* camera = renderer_->GetActiveCamera();
  camera->SetClippingRange(0.1, 1e+10);
  camera->SetPosition(0., -1000., 0.);
  double pos[3] = {0., 0., 0.};
  camera->SetEyePosition(pos);
  camera->SetViewUp(0., 0., 1.);
  renderer_->ResetCamera();
  render_window_->Render();
  return;
}

bool QMapViewer::eventFilter(QObject* watched, QEvent *event) {
  if(watched == qvtk_widget_ && event->type() ==  QEvent::KeyPress){
    QKeyEvent* ke = static_cast<QKeyEvent*>(event);
    QMapViewer::keyPressEvent(ke);
    return true;
  }
  return false;
}

void QMapViewer::Save() {
  std::cout << "Save... ";
  pipeline_->Save();
  std::cout << "Done" << std::endl;
  return;
}

void EvaluateCovisibility(Frame* kf){
  // TODO mappoint->keyframes의 median 계산.
  const std::set<Mappoint*> mappoints = kf->GetMappoints();
  std::vector<size_t> n_keyframes;
  for(Mappoint* mp : mappoints){
    if(mp->GetRefFrame() == kf)
      continue;
    size_t n = mp->GetKeyframes().size();
    n_keyframes.push_back(n);
  }

  if(n_keyframes.size()<4)
    return;

  std::sort(n_keyframes.begin(), n_keyframes.end());

  int mid_index = n_keyframes.size()/2;
  std::cout << "max/median n(mp->keyframes()) = "
    << n_keyframes.at(n_keyframes.size()-1) << ", "
    << n_keyframes.at(mid_index) 
    << ", from n(mp) = " << n_keyframes.size()
    << std::endl;
  return;
}


void QMapViewer::UpdateMappoints(Frame* frame){
  return; // TODO config drawing mappoints.
  const auto mappoints = frame->GetMappoints();
  for(Mappoint* mp : mappoints){
    if(!mp)
      continue;
    auto Xw = mp->GetXw();
    if(drawed_mappoints_.count(mp)){
      points_->SetPoint(drawed_mappoints_.at(mp),
                        Xw.x(), Xw.y(), Xw.z());
    }
    else{
      drawed_mappoints_[mp] = points_->GetNumberOfPoints();
      points_->InsertNextPoint(Xw.x(), Xw.y(), Xw.z());
    }
  }
  vertexglyphfilter_->Modified();
  mappoints_actor_->Modified();
}

void QMapViewer::OnSetKeyframe(Frame* kf) {
  EvaluateCovisibility(kf);

  std::map<int, Frame*> frames = pipeline_->GetMap()->GetFrames();
  EigenMap<int, g2o::SE3Quat> Tcws;
  for(auto it : frames){
    if(!it.second->IsKeyframe())
      continue;
    Tcws[it.first] = it.second->GetTcw();
  }
  est_drawer_.Draw(Tcws);

  auto actor_est_trj = est_drawer_.GetActor();
	actor_est_trj->GetProperty()->SetLineWidth(2);
	actor_est_trj->GetProperty()->SetColor(0., 0.8, 0.);
  renderer_->AddActor(actor_est_trj);

  if(dataset_){
    Tgt_est_
      = AlignTrajectory(dataset_->GetTcws(), frames, Tgt_est_);
    SetActorTf(gt_drawer_.GetActor(), Tgt_est_.inverse());
  }
  return;
}

void QMapViewer::OnFrame(Frame* frame, const FrameInfo& info) {
  g2o::SE3Quat Tcw = frame->GetTcw();
  SetActorTf(coord_actor_, Tcw.inverse());
  {
    size_t n = frame->GetIndex();
    std::string msg = std::to_string(n+1) + "/" + std::to_string(dataset_->Size());
    corner_annotation_->SetText(0, msg.c_str());
  }
  {
    std::ostringstream out;
    out.precision(1);
    out << std::fixed << info.elapsed_ms_;
    std::string msg = out.str() + "[ms/frame]";
    corner_annotation_->SetText(1, msg.c_str());
  }
  render_window_->Render();
}

void QMapViewer::keyPressEvent(QKeyEvent *event) {
  switch(event->key()){
    case Qt::Key_S:
      Save();
      break;
    case Qt::Key_P:
      if(pipeline_player_->IsActive())
        pipeline_player_->Stop();
      else
        pipeline_player_->Start();
      break;
    case Qt::Key_Q:
      QMapViewer::close();
      break;
  }
  return;
}

CvViewer::CvViewer(Pipeline* pipeline)
: pipeline_(pipeline){
  pipeline->AddViewer(this);
}

void CvViewer::OnFrame(Frame* frame, const FrameInfo& info){
  if(frame->GetImage().empty())
    return;

  PipelineMap* map = pipeline_->GetMap();
  map->Lock();
  // vsiualize
  cv::Mat dst;
  cv::cvtColor(frame->GetImage(), dst, cv::COLOR_GRAY2RGB);
  auto keypoints = frame->GetKeypoints();
  auto vec_mappoitns = frame->GetVecMappoints();
  for(size_t i = 0; i < keypoints.size(); i++){
    Mappoint* mp = vec_mappoitns.at(i);
    cv::circle(dst, keypoints.at(i).pt, 2,
               mp?CV_RGB(0,255,0):CV_RGB(255,0,0));
    if(!mp)
      continue;
#if 0
    auto keyframes = mp->GetKeyframes(frame->GetIndex());
    std::map<int, Frame*> ordered_kfs;
    for(Frame* kf : keyframes)
      ordered_kfs[kf->GetIndex()] = kf;
    cv::KeyPoint kpt0;
    {
      auto it0 = ordered_kfs.begin();
      Frame* f0 = it0->second;
      kpt0 = f0->GetKeypoint(mp);
    }
    for(auto it = ordered_kfs.begin(), it_end = ordered_kfs.end(); ; it++){
      auto it1 = it;
      if(++it1 == it_end)
        break;
      Frame* f1 = it->second;
      auto kpt1 = f1->GetKeypoint(mp);
      cv::line(dst, kpt0.pt, kpt1.pt, CV_RGB(255,255,0),2);
      kpt0 = kpt1;
    }
#else
    const cv::KeyPoint& kpt = keypoints.at(i);
    Frame* ref = mp->GetRefFrame();
    int idx0 = ref->GetIndex(mp);
    const cv::KeyPoint& kpt0 = ref->GetKeypoints().at(idx0);
    cv::line(dst, kpt.pt, kpt0.pt, CV_RGB(0,0,255), 1);
#endif
  }

  map->UnLock();
  cv::imshow("track",dst);
  char c = cv::waitKey(1);
  return;
}
