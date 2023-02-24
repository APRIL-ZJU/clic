/*
 * Continuous-Time Fixed-Lag Smoothing for LiDAR-Inertial-Camera SLAM
 * Copyright (C) 2022 Jiajun Lv
 * Copyright (C) 2022 Xiaolei Lang
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#pragma once

#include <geometry_msgs/PointStamped.h>
#include <sensor_msgs/PointCloud.h>  // camera feature

#include <spline/trajectory.h>
#include <utils/tic_toc.h>
#include <utils/yaml_utils.h>

#include "feature_manager.h"
#include "vio_initial.h"

namespace clic {

enum MarginalizationFlag {
  MARGIN_OLD = 0,  //
  MARGIN_SECOND_NEW = 1
};

class VisualOdometry : public FeatureManager {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef std::shared_ptr<VisualOdometry> Ptr;
  using Vector7D = Eigen::Matrix<double, 7, 1>;

  VisualOdometry(const YAML::Node& node, Trajectory::Ptr traj)
      : trajectory_(std::move(traj)) {
    readParameters(node);
    ClearState();
  }

  // initial landmarks
  void AddImageToWindow(const sensor_msgs::PointCloud::ConstPtr& img_msg);

  void SlideWindow();

  bool IsWindowOk() const { return (frame_count_ == WINDOW_SIZE); }

  ///================== get functions ================== ///

  MarginalizationFlag GetMarginalizationFlag() { return marg_flag_; }

  const double* GetTimestamps() const { return timestamps_; }

  const std::list<FeaturePerId>& GetFeatures() const { return this->feature; }

  const std::vector<int>& GetGoodFeatureIDs() const {
    return good_feature_ids_;
  }

  double GetReprojectionError(const FeaturePerId& feature_checked,
                              const Eigen::Vector3d& p_inG,
                              const Eigen::Vector3d Ps_cam[],
                              const Eigen::Matrix3d Rs_cam[],
                              std::pair<int, int> project_frame) const;

  bool TriangulateLandmarks(bool has_new_traj);

  void SetDepth(const std::map<int, double> fea_id_inv_depths);

  bool GetCurOptLandmarks(VPointCloud& landmarks_init,
                          VPointCloud& landmarks_opt) const;

 private:
  static bool IsLandMarkStable(const FeaturePerId& fea) {
    if (!FeatureManager::isLandmarkCandidate(fea)) return false;
    if (fea.start_frame > WINDOW_SIZE * 3.0 / 4.0) return false;
    if (fea.estimated_depth <= 0) return false;
    // TODO
    // if (it_per_id.solve_flag != SovelSucc)
    return true;
  }

  void ClearState();

  void SlideWindowNew();

  void SlideWindowOld();

 private:
  Trajectory::Ptr trajectory_;

  MarginalizationFlag marg_flag_;

  int frame_count_;

  // image timestamps
  double timestamps_[(WINDOW_SIZE + 1)];

  double marg_old_t0_;

  std::vector<int> good_feature_ids_;
  double last_landmark_time;
};

}  // namespace clic
