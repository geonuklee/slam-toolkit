#ifndef SEG_DATASET_H_
#define SEG_DATASET_H_
#include "segslam.h"
#include "stdafx.h"
#include "camera.h"

void WriteKittiTrajectory(const g2o::SE3Quat& Tcw, std::ofstream& output_file);

class KittiTrackingDataset {
public:
  KittiTrackingDataset(std::string type, //  "training"|"testing"
                       std::string seq,
                       std::string dataset_path="");
  virtual ~KittiTrackingDataset();

  cv::Mat GetImage(int i, int flags=cv::IMREAD_COLOR) const;
  cv::Mat GetRightImage(int i, int flags=cv::IMREAD_COLOR) const; 
  cv::Mat GetDepthImage(int i) const;
  cv::Mat GetDynamicMask(int i) const;
  cv::Mat GetInstanceMask(int i) const;
  double GetSecond(int i) const;

  bool EixstCachedDepthImages() const;
  void ComputeCacheImages(); // depth image, dynamic instance mask.

  virtual int Size() const;
  virtual const EigenMap<int, g2o::SE3Quat>& GetTcws() const;
  virtual const Camera* GetCamera() const;
private:
  std::string dataset_path_;
  std::string type_;
  std::string seq_;
  std::string depth_path_, mask_path_;

  std::map<int, std::string> im0_filenames_;
  std::map<int, std::string> im1_filenames_;
  EigenMap<int, g2o::SE3Quat> Tcws_;
  Camera* camera_; // TODO Steroe, Depth, etc..
};

namespace NEW_SEG {
class Frame;
class EvalWriter {
public:
  EvalWriter(std::string dataset_type, std::string seq, std::string output_dir);
  void Write(Frame* frame,
             RigidGroup* static_rig,
             const cv::Mat synced_marker,
             const cv::Mat gt_insmask,
             const cv::Mat gt_dmask
             );
  void WriteInstances(const std::map<Pth, Pth>& pth_removed2replacing);

private:
  int start_frame_;
  int last_frame_;
  std::map<int, int> pth2qth_; // Update from Write()
  std::string seq_;
  std::string output_seq_dir_;
  std::string output_mask_dir_;

  std::ofstream trj_output_;

  std::ofstream keypoints_output_;
  std::ofstream instances_output_;

  std::ofstream seqmap_output_;
  std::ofstream kitti2dbox_output_;
};

};

#endif
