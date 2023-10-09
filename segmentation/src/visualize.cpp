#include "segslam.h"
#include "util.h"
#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>

namespace NEW_SEG {

cv::Mat VisualizeSwitchStates(RigidGroup* rig,
                              const std::map<Jth,Frame*>& keyframes,
                              const std::map<Pth, float>& switch_states,
                              const std::map<Pth, std::pair<cv::Point2f, float> >& pth2center,
                              cv::Size dst_size) {
  Qth qth = rig->GetId();
  cv::Mat dst_texts = cv::Mat(dst_size.height, dst_size.width, CV_8UC3); 
  int fontFace = cv::FONT_HERSHEY_SIMPLEX;
  double fontScale = .4;
  int thickness = 1;
  int baseline=0;
  int y = 0;
  cv::rectangle(dst_texts,cv::Rect(0,0,dst_texts.cols, dst_texts.rows),CV_RGB(255,255,255),-1);
  for(auto it_kf : keyframes){
    const Jth& jth = it_kf.first;
    Frame* kf = it_kf.second;
    char text[100];
    sprintf(text,"F#%2d, KF#%2d ", kf->GetId(), kf->GetKfId(qth) ); // translation 등의 정보는 Pangolin viewer로 충분.
    y += 2;
  }
  y+= 3;
  std::ostringstream buffer;
  buffer << "Exclude> ";
  for(auto it : rig->GetExcludedInstances()){
    const Pth& pth = it.first;
    if(!pth2center.count(pth)) // TODO 현재 화면 보이는 exclude만 표시할까?
      continue;
    buffer << pth << ", ";
    auto size = cv::getTextSize(buffer.str(), fontFace, fontScale, thickness, &baseline);
    if(size.width < dst_texts.cols - 10)
      continue;
    y += size.height+2;
    cv::putText(dst_texts, buffer.str(), cv::Point(5, y), fontFace, fontScale, CV_RGB(0,0,0), thickness);
    buffer.str("");
  }
  if(!buffer.str().empty()){
    auto size = cv::getTextSize(buffer.str(), fontFace, fontScale, thickness, &baseline);
    y += size.height+2;
    cv::putText(dst_texts, buffer.str(), cv::Point(5, y), fontFace, fontScale, CV_RGB(0,0,0), thickness);
  }

  y+= 10;
  buffer.str("");
  for(auto it : switch_states){
    buffer << "(" << it.first <<  ":" << std::fixed << std::setprecision(3) << it.second << "), ";
    auto size = cv::getTextSize(buffer.str(), fontFace, fontScale, thickness, &baseline);
    if(size.width < dst_texts.cols - 100)
      continue;
    y += size.height+2;
    cv::putText(dst_texts, buffer.str(), cv::Point(5, y), fontFace, fontScale, CV_RGB(0,0,0), thickness);
    buffer.str("");
  }
  if(!buffer.str().empty()){
    auto size = cv::getTextSize(buffer.str(), fontFace, fontScale, thickness, &baseline);
    y += size.height+2;
    cv::putText(dst_texts, buffer.str(), cv::Point(5, y), fontFace, fontScale, CV_RGB(0,0,0), thickness);
  }

  return dst_texts;
}

void PutInstanceInfoBox(const float& switch_threshold,
                        const std::map<Pth, float>& switch_states,
                        const std::map<Pth,float>& density_scores,
                        const std::map<Pth, std::pair<cv::Point2f, float> >& pth2center,
                        cv::Mat& dst){
  for(auto it : pth2center){
    const Pth& pth = it.first;
    const cv::Point2f& cp = it.second.first;
    std::string msg[3];
    cv::Scalar txt_colors[3];
    msg[0] = "#" + std::to_string(pth);
    for(int k=0; k<3;k++)
      txt_colors[k] = CV_RGB(0,0,0);
    msg[1] = "d: "; msg[2] = "s: ";
    if(density_scores.count(pth) ){
      std::ostringstream oss; oss << std::fixed << std::setprecision(2);
      oss<< density_scores.at(pth);
      msg[1] += oss.str();
      if(density_scores.at(pth) < 1.)
        txt_colors[1] = CV_RGB(255,0,0);
    }
    else
      msg[1] += "-";
    if(switch_states.count(pth) ){
      std::ostringstream oss; oss << std::fixed << std::setprecision(2);
      oss<< switch_states.at(pth);
      msg[2] += oss.str();
      if(switch_states.at(pth) < switch_threshold)
        txt_colors[2] = CV_RGB(255,0,0);
    }
    else
      msg[2] += "-";
    int offset = 2;
    int w, h;
    w = h = 0;
    int fontFace = cv::FONT_HERSHEY_SIMPLEX; double fontScale = .3; int fontThick = 1; int baseline = 0;
    for(int k=0; k<3; k++){
      auto size = cv::getTextSize(msg[k], fontFace, fontScale, fontThick, &baseline);
      h += size.height;
      w = std::max(size.width, w);
    }
    cv::Point2f dpt(.5*w, .5*h);
    cv::rectangle(dst, cp-dpt-cv::Point2f(0,3*baseline), cp+dpt, CV_RGB(255,255,255), -1);
    int x = cp.x - .5*w;
    int y = cp.y - .5*h;
    for(int k=0;k<3;k++){
      cv::putText(dst, msg[k], cv::Point(x,y),fontFace, fontScale, txt_colors[k], fontThick);
      auto size = cv::getTextSize(msg[k], fontFace, fontScale, fontThick, &baseline);
      y += size.height+offset;
    }
  }
  return;
}


void Pipeline::Visualize(const cv::Mat rgb, const cv::Mat gt_dynamic_mask, cv::Mat& dst) {
  Qth qth = 0;
  RigidGroup* rig = qth2rig_groups_.at(qth);
  cv::Mat dst_frame = rgb.clone();
  std::map<Pth, std::pair<cv::Point2f, float> > pth2center;

  cv::Mat outline = vinfo_synced_marker_ < 1;
  cv::Mat dist_from_outline;
  {
    cv::Mat bb = ~outline;
    for (int i = 0; i < bb.cols; i++) {
      bb.at<uchar>(0, i) = bb.at<uchar>(bb.rows - 1, i) = 0;
    }
    for (int i = 0; i < bb.rows; i++) {
      bb.at<uchar>(i, 0) = bb.at<uchar>(i, bb.cols - 1) = 0;
    }
    cv::distanceTransform(bb,dist_from_outline, cv::DIST_L2, cv::DIST_MASK_3);
  }

  std::set<Pth> excluded;
  if(vinfo_switch_states_.count(qth)){
    RigidGroup* rig = qth2rig_groups_.at(qth);
    const auto& switch_states = vinfo_switch_states_.at(qth);
    for(auto it : rig->GetExcludedInstances())
      excluded.insert(it.first);
    //for(auto it : switch_states){
    //  if(it.second > switch_threshold_)
    //    continue;
    //  excluded.insert(it.first);
    //}
  }
  for(size_t i = 0; i < dst_frame.rows; i++){
    for(size_t j = 0; j < dst_frame.cols; j++){
      cv::Vec3b& pixel = dst_frame.at<cv::Vec3b>(i,j);
      if(outline.at<uchar>(i,j))
        continue;
      const Pth& pth = vinfo_synced_marker_.at<int32_t>(i,j);
      const float& r = dist_from_outline.at<float>(i,j);
      std::pair<cv::Point2f, float>& cp = pth2center[pth];
      if(r > cp.second){
        cp.first = cv::Point2f(j,i);
        cp.second = r;
      }
    }
  }
 dst_frame.setTo(CV_RGB(0,0,0), outline);
  for(Pth pth : excluded)
    dst_frame.setTo(CV_RGB(255,0,0), vinfo_synced_marker_==pth);
  if(vinfo_switch_states_.count(qth)){
    for(auto it : vinfo_switch_states_.at(qth)){
      if(it.second > switch_threshold_)
        continue;
      dst_frame.setTo(CV_RGB(255,255,0), vinfo_synced_marker_==it.first);
    }
  }
  cv::addWeighted(rgb, .5, dst_frame, .5, 1., dst_frame);

  const auto& keypoints = prev_frame_->GetKeypoints();
  const auto& mappoints = prev_frame_->GetMappoints();
  const auto& instances = prev_frame_->GetInstances();
  for(size_t n=0; n<keypoints.size(); n++){
    cv::Point2f pt = keypoints[n].pt;
    Mappoint* mp = mappoints[n];
    if(mp)
      continue;
    cv::circle(dst_frame, pt, 3, CV_RGB(150,150,150), 1);
  }

  for(size_t n=0; n<keypoints.size(); n++){
    cv::Point2f pt = keypoints[n].pt;
    Mappoint* mp = mappoints[n];
    if(!mp)
      continue;
    const bool& supplied_pt = vinfo_supplied_mappoints_[n];
    if(supplied_pt)
      cv::circle(dst_frame, pt, 3, CV_RGB(255,0,0), 1);
  }
  std::list<Mappoint*> sorted_mappoints;
  for(size_t n=0; n<keypoints.size(); n++){
    cv::Point2f pt = keypoints[n].pt;
    Mappoint* mp = mappoints[n];
    if(!mp)
      continue;
    const bool& supplied_pt = vinfo_supplied_mappoints_[n];
    if(supplied_pt)
      continue;
    Instance* ins = instances[n];
    // Instance* ins = m->GetInstance();
    sorted_mappoints.push_back(mp);
    cv::circle(dst_frame, pt, 3, CV_RGB(0,255,0), -1);
    const std::set<Frame*>& keyframes = mp->GetKeyframes(qth);
    for(auto it = keyframes.rbegin(); it!=keyframes.rend(); it++){
      const cv::Point2f& pt_next = (*it)->GetKeypoint( (*it)->GetIndex(mp) ).pt;
      cv::line(dst_frame, pt, pt_next, CV_RGB(0,0,255), 1);
      if(*it != prev_frame_)
        break;
    }
  }

  sorted_mappoints.sort([](Mappoint* a, Mappoint* b) {
                        Pth pa = a->GetInstance()->GetId();
                        Pth pb = b->GetInstance()->GetId();
                        if(pa != pb)
                          return pa < pb;
                        return a->GetId() < b->GetId();
                        });

  // Draw patches.
  cv::Size patch_size(20,20);
  int n_rows = 5;
  int n_cols = 10;
  cv::Mat dst_patches = cv::Mat::zeros(patch_size.height*n_rows, patch_size.width*n_cols, CV_8UC3);
  dst_patches.setTo(CV_RGB(200,0,255));
  {
    static std::map<Jth, cv::Mat> gt_masks;
    gt_masks[prev_frame_->GetId()] = gt_dynamic_mask;
    int i_col = 0;
    for(Mappoint* mp : sorted_mappoints){
      Frame* ref = mp->GetRefFrame(0);
      cv::Point2f pt_ref = ref->GetKeypoint(ref->GetIndex(mp)).pt;
      if(!gt_masks[ref->GetId()].at<uchar>(pt_ref))
        continue;
      Pth pth = mp->GetInstance()->GetId();
      //if(mp->GetInstance()->GetId() != 101)
      //  continue;
      std::map<int, Frame*> sorted_keyframes;
      for(Frame* kf : mp->GetKeyframes(qth))
        sorted_keyframes[kf->GetId()] = kf;
      sorted_keyframes[prev_frame_->GetId()] = prev_frame_;
      int i_row = 0;
      for(auto it_kf : sorted_keyframes){
        Frame* kf = it_kf.second;
        const cv::Point2f& pt = kf->GetKeypoint( kf->GetIndex(mp) ).pt;
        const cv::Mat& rgb = kf->GetRgb();
        if(rgb.empty())
          continue;
        int x = static_cast<int>(pt.x - int(patch_size.width/2));
        int y = static_cast<int>(pt.y - int(patch_size.height/2));
        if (x < 0 || y < 0 || x + patch_size.width >= rgb.cols || y + patch_size.height >= rgb.rows) 
          continue;
        cv::Rect rgb_roi(x, y, patch_size.width, patch_size.height);
        cv::Rect dst_roi(patch_size.width*i_col, patch_size.height*i_row,patch_size.width, patch_size.height);
        rgb(rgb_roi).copyTo(dst_patches(dst_roi));
        {
          cv::Point pt(patch_size.width*i_col, patch_size.height*i_row);
          cv::putText(dst_patches, std::to_string(kf->GetId()%100),pt, cv::FONT_HERSHEY_SIMPLEX, .4, CV_RGB(200,0,255) );
        }

        if(i_row==0){
          cv::Point pt(patch_size.width*i_col, dst_patches.rows-2);
          cv::putText(dst_patches, std::to_string(pth%100),pt, cv::FONT_HERSHEY_SIMPLEX, .5, CV_RGB(0,0,0) );
        }

        if(++i_row >= n_rows)
          break;
      }
      if(++i_col >= n_cols)
        break;
    }
  }
  
  const auto& switch_states = vinfo_switch_states_[qth];
  const auto& density_scores = vinfo_density_socres_;
  cv::Mat dst_marker = GetColoredLabel(vinfo_synced_marker_);
  PutInstanceInfoBox(switch_threshold_, switch_states, density_scores, pth2center, dst_marker);

  std::map<Jth, Frame*> keyframes = vinfo_neighbor_frames_[qth];
  keyframes[prev_frame_->GetId()] = prev_frame_;
  cv::Mat dst_switchstates = VisualizeSwitchStates(rig, keyframes, switch_states, pth2center, cv::Size(100,600) );

  {
    int fid = prev_frame_->GetId();
    int kfid = prev_frame_->GetKfId(qth);
    std::string msg = "KF#" + std::to_string(kfid) + "/ F#" + std::to_string(fid);
    int fontFace = cv::FONT_HERSHEY_SIMPLEX; double fontScale = .6; int fontThick = 1; int baseline = 0;
    cv::Size size = cv::getTextSize(msg, fontFace, fontScale, fontThick, &baseline);
    cv::rectangle(dst_marker, cv::Point(0,0), cv::Point(size.width+10,size.height+10), CV_RGB(150,150,150), -1);
    cv::putText(dst_marker, msg, cv::Point(5,size.height+5),fontFace, fontScale, CV_RGB(255,0,0), fontThick);
  }

  cv::vconcat(dst_marker,dst_frame,dst);
  //cv::imshow("switchstates", dst_switchstates);
  //cv::imshow("dst", dst);
  cv::imshow("dpatches", dst_patches);
  cv::moveWindow("dpatches", 10, 900);
  return;
} // Pipeline::Visualize

} // namespace NEW_SEG
