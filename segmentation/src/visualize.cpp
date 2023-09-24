#include "segslam.h"
#include "util.h"
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

namespace OLD_SEG {

void Pipeline::Visualize(const cv::Mat vis_rgb,
                         const EigenMap<int, g2o::SE3Quat>* gt_Tcws) {
  cv::Mat rgb = vis_rgb;
  cv::Mat dst_frame = rgb.clone();
  cv::Mat outline = vinfo_synced_marker_ < 1;
  if(prev_frame_->GetId() == 0){
    // 첫번째 frame이다.
    return;
  }

  cv::Mat dist_from_outline;
  cv::distanceTransform(~outline,dist_from_outline, cv::DIST_L2, cv::DIST_MASK_3);
  Qth dominant_qth = 0;
  RigidGroup* dominant_rig = qth2rig_groups_.at(dominant_qth);

  Qth qth = dominant_qth;
  std::set<Pth> excluded; {
    RigidGroup* rig = qth2rig_groups_.at(qth);
    const auto& switch_states = vinfo_switch_states_.at(qth);
    for(auto it : rig->GetExcludedInstances())
      excluded.insert(it.first);
    for(auto it : switch_states){
      if(it.second > switch_threshold_)
        continue;
      excluded.insert(it.first);
    }
  }

  std::map<Pth, std::pair<cv::Point2f, float> > pth2center;
  for(size_t i = 0; i < dst_frame.rows; i++){
    for(size_t j = 0; j < dst_frame.cols; j++){
      cv::Vec3b& pixel = dst_frame.at<cv::Vec3b>(i,j);
      if(outline.at<uchar>(i,j)){
        pixel[0] = pixel[1] =pixel[2] = 0;
        continue;
      }
      const Pth& pth = vinfo_synced_marker_.at<int32_t>(i,j);
      const float& r = dist_from_outline.at<float>(i,j);
      std::pair<cv::Point2f, float>& cp = pth2center[pth];
      if(r > cp.second){
        cp.first = cv::Point2f(j,i);
        cp.second = r;
      }
      if(excluded.count(pth)){
        pixel[0] = pixel[1] = 0;
        pixel[2] = 255;
      }
    }
  }
  cv::addWeighted(rgb, .5, dst_frame, .5, 1., dst_frame);
  

  const auto& keypoints = prev_frame_->GetKeypoints();
  const auto& mappoints = prev_frame_->GetMappoints();
  const auto& instances = prev_frame_->GetInstances();
  for(size_t n=0; n<keypoints.size(); n++){
    Instance* ins = instances[n];
    Mappoint* mp = mappoints[n];
    const cv::Point2f& pt = keypoints[n].pt;
    if(mp){
      std::stringstream ss;
      ss << std::hex << mp->GetId(); // Convert to hexadecimal
      std::string msg = ss.str();
      if(mp->HasEstimate4Rig(dominant_qth)) {
        const std::set<Frame*>& keyframes = mp->GetKeyframes(dominant_qth);
        for(auto it = keyframes.rbegin(); it!=keyframes.rend(); it++){
          const cv::Point2f& pt0 = (*it)->GetKeypoint( (*it)->GetIndex(mp) ).pt;
          cv::circle(dst_frame, pt, 3, CV_RGB(0,255,0), -1);
          cv::line(dst_frame, pt, pt0, CV_RGB(0,0,255),  2);
          //break;
        }
      }
      else
        cv::circle(dst_frame, pt, 3, CV_RGB(255, 0,0), -1);
    }
    else {
      cv::circle(dst_frame, pt, 2, CV_RGB(150,150,150), 1);
    }
  }

  {
    const auto& density_scores = vinfo_density_socres_;
    const auto& switch_states = vinfo_switch_states_.at(qth);
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
        if(switch_states.at(pth) < switch_threshold_)
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
      cv::rectangle(dst_frame, cp-dpt-cv::Point2f(0,3*baseline), cp+dpt, CV_RGB(255,255,255), -1);
      int x = cp.x - .5*w;
      int y = cp.y - .5*h;
      for(int k=0;k<3;k++){
        cv::putText(dst_frame, msg[k], cv::Point(x,y),fontFace, fontScale, txt_colors[k], fontThick);
        auto size = cv::getTextSize(msg[k], fontFace, fontScale, fontThick, &baseline);
        y += size.height+offset;
      }
    }
  }

  cv::Mat dst_texts = cv::Mat::zeros(rgb.rows*2, 400, CV_8UC3); {
    const auto& switch_states = vinfo_switch_states_.at(dominant_qth);
    int fontFace = cv::FONT_HERSHEY_SIMPLEX;
    double fontScale = .4;
    int thickness = 1;
    int baseline=0;
    int y = 0;
    cv::rectangle(dst_texts,cv::Rect(0,0,dst_texts.cols, dst_texts.rows),CV_RGB(255,255,255),-1);
    std::map<Jth, Frame*> keyframes = vinfo_neighbor_frames_.at(qth);
    keyframes[prev_frame_->GetId()] = prev_frame_;
    Frame* prev_kf = nullptr;
    for(auto it_kf : keyframes){
      const Jth& jth = it_kf.first;
      Frame* kf = it_kf.second;
      auto t = kf->GetTcq(dominant_qth).inverse().translation();
      char text[100];

      sprintf(text,"est : F#%2d, KF#%2d  (%4.3f, %4.3f, %4.3f)", kf->GetId(), kf->GetKfId(dominant_qth), t[0], t[1], t[2] );
      auto size = cv::getTextSize(text, fontFace, fontScale, thickness, &baseline);
      y += size.height+2;
      cv::putText(dst_texts, text, cv::Point(5, y), fontFace, fontScale, CV_RGB(0,0,0), thickness);

      if(!gt_Tcws)
        continue;
      const g2o::SE3Quat& Tcw = gt_Tcws->at(jth);
      t = Tcw.inverse().translation();
      sprintf(text,"true: F#%2d,  (%4.3f, %4.3f, %4.3f)", kf->GetId(), t[0], t[1], t[2] );
      size = cv::getTextSize(text, fontFace, fontScale, thickness, &baseline);
      y += size.height+2;
      cv::putText(dst_texts, text, cv::Point(5, y), fontFace, fontScale, CV_RGB(120,120,120), thickness);

      if(!prev_kf){
        prev_kf = kf;
        continue;
      }
      // dT =Tc0 c1 = Tc0 w * Tc1 w.inverse()
      g2o::SE3Quat est_dT = prev_kf->GetTcq(dominant_qth) * kf->GetTcq(dominant_qth).inverse();
      g2o::SE3Quat true_dT = gt_Tcws->at(prev_kf->GetId()) * gt_Tcws->at(kf->GetId()).inverse();
      g2o::SE3Quat err_dT = true_dT.inverse() * est_dT;
      if(true_dT.translation().norm() < 1e-2)
        continue;
      float norm_ratio = est_dT.translation().norm() / true_dT.translation().norm();
      float err_ratio = err_dT.translation().norm() / true_dT.translation().norm();
      sprintf(text,"norm ratio: %4.3f", norm_ratio);
      cv::putText(dst_texts, text, cv::Point(20+size.width, y), fontFace, fontScale, CV_RGB(255,0,0), thickness);
      y += 2;
      prev_kf = kf;
    } // for keyframes

    y+= 10;
    std::ostringstream buffer;
    buffer << "Include> ";
    for(auto it : dominant_rig->GetIncludedInstances()){
      const Pth& pth = it.first;
      if(!pth2center.count(pth))
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
    y+= 3;
    buffer.str("");
    buffer << "Exclude> ";
    for(auto it : dominant_rig->GetExcludedInstances()){
      const Pth& pth = it.first;
      if(!pth2center.count(pth))
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
    buffer << "Pth:Switch state> ";
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
  }

  {
    int fontFace = cv::FONT_HERSHEY_SIMPLEX; double fontScale = .8; int fontThick = 2; int baseline = 0;
    std::string msg = "F#" + std::to_string(prev_frame_->GetId()) + (prev_frame_->IsKeyframe() ? ", KF" : "");
    cv::putText(dst_frame, msg, cv::Point(10,20),fontFace, fontScale, CV_RGB(255,0,0), fontThick);
  }

  cv::imshow("dst_frame", dst_frame);
  cv::imshow("dst_text", dst_texts);
  //cv::imshow("sync_marker", GetColoredLabel(vinfo_synced_marker_));
  return;
}

} // namespace OLD_SEG

namespace NEW_SEG {

void Pipeline::Visualize(const cv::Mat rgb) {
  Qth qth = 0;
  Frame* kf_prev = nullptr;{
    auto qth_keyframes = keyframes_.at(qth);
    for(auto it_kf = qth_keyframes.rbegin(); it_kf != qth_keyframes.rend(); it_kf++){
      if(it_kf->second == prev_frame_)
        continue;
      kf_prev = it_kf->second;
      break;
    }
  }
  if(kf_prev){
    cv::Mat dst_kf0 = kf_prev->GetRgb().clone();
    const auto& keypoints = kf_prev->GetKeypoints();
    const auto& mappoints = kf_prev->GetMappoints();
    const auto& instances = kf_prev->GetInstances();
    for(size_t n=0; n<keypoints.size(); n++){
      cv::Point2f pt = keypoints[n].pt;
      Mappoint* mp = mappoints[n];
      if(!mp)
        cv::circle(dst_kf0, pt, 3, CV_RGB(150,150,150), 1);
      else
        cv::circle(dst_kf0, pt, 3, CV_RGB(0,255,0), -1);
    }
    cv::imshow("kf0", dst_kf0);
  }

  cv::Mat dst_curr = rgb.clone();
  const auto& keypoints = prev_frame_->GetKeypoints();
  const auto& mappoints = prev_frame_->GetMappoints();
  const auto& instances = prev_frame_->GetInstances();
  for(size_t n=0; n<keypoints.size(); n++){
    cv::Point2f pt = keypoints[n].pt;
    Instance* ins = instances[n];
    Mappoint* mp = mappoints[n];
    if(!mp){
      cv::circle(dst_curr, pt, 3, CV_RGB(150,150,150), 1);
      continue;
    }
    //const bool& supplied_pt = vinfo_supplied_mappoints_[n];
    //if(supplied_pt){
    //  cv::circle(dst_curr, pt, 3, CV_RGB(0,255,0), 1);
    //  continue;
    //}
    cv::circle(dst_curr, pt, 3, CV_RGB(0,255,0), -1);
    const std::set<Frame*>& keyframes = mp->GetKeyframes(qth);
    for(auto it = keyframes.rbegin(); it!=keyframes.rend(); it++){
      const cv::Point2f& pt_next = (*it)->GetKeypoint( (*it)->GetIndex(mp) ).pt;
      cv::line(dst_curr, pt, pt_next, CV_RGB(0,0,255),  2);
    }
  }
  cv::imshow("dst_curr", dst_curr);
  return;
}

} // namespace NEW_SEG
