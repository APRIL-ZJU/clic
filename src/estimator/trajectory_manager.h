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
#include <estimator/trajectory_estimator.h>
#include <inertial/imu_state_estimator.h>
#include <spline/trajectory.h>
#include <utils/log_utils.h>

#include <estimator/factor/analytic_diff/marginalization_factor.h>
#include <utils/opt_weight.h>

#include <visual_odometry/integration_base.h>
#include <visual_odometry/visual_odometry.h>
#include <visual_odometry/visual_struct.h>

namespace clic {

struct TimeParam {
  TimeParam() {
    traj_active = -1;
    for (int i = 0; i < 2; ++i) {
      lio_imu_time[i] = -1;
      lio_imu_idx[i] = 0;

      lio_map_imu_time[i] = -1;
      lio_map_imu_idx[i] = 0;

      last_scan[i] = -1;
      cur_scan[i] = -1;
      visual_window[i] = -1;
    }
  }

  void UpdateCurScan(double scan_time_min, double scan_time_max) {
    last_scan[0] = cur_scan[0];
    last_scan[1] = cur_scan[1];

    cur_scan[0] = scan_time_min;
    cur_scan[1] = scan_time_max;
  }

  double lio_imu_time[2];  // the time of imu involved in this optimiazation
  int lio_imu_idx[2];      // the index of imu involved in this optimiazation

  double lio_map_imu_time[2];
  int lio_map_imu_idx[2];

  double traj_active;       // the ctrl points before that is fixed (lio)
  double last_scan[2];      // time of last scan
  double cur_scan[2];       // time of current scan
  double visual_window[2];  // time of visual window
};

class TrajectoryManager {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef std::shared_ptr<TrajectoryManager> Ptr;

  TrajectoryManager(const YAML::Node& node, Trajectory::Ptr trajectory);

  void InitFactorInfo(
      const ExtrinsicParam& Ep_CtoI, const ExtrinsicParam& Ep_LtoI,
      const double image_feature_weight = 0);

  void SetTrajectory(Trajectory::Ptr trajectory) { trajectory_ = trajectory; }

  void SetSystemState(const SystemState& sys_state);

  void SetOriginalPose(Eigen::Quaterniond q,
                       Eigen::Vector3d p = Eigen::Vector3d::Zero());

  void AddIMUData(const IMUData& data);

  size_t GetIMUDataSize() const { return imu_data_.size(); }

  void SetUpdatedLoop() { updated_loop_ = true; }

  void PropagateTrajectory(double scan_time_min, double scan_time_max);

  void UpdateLIOPrior(
      const Eigen::aligned_vector<PointCorrespondence>& point_corrs);

  void UpdateVisualOffsetPrior(const std::list<FeaturePerId>& features,
                               const std::vector<int>& good_feature_ids,
                               const double timestamps[]);

  // clear the prior after loop closure
  void ClearLIOPrior() {
    lidar_marg_info = nullptr;
    lidar_prior_ctrl_id = std::make_pair(0, 0);
    lidar_marg_parameter_blocks.clear();
  }

  bool UpdateTrajectoryWithLoamFeature(
      const Eigen::aligned_vector<PointCorrespondence>& point_corrs,
      const std::list<FeaturePerId>& features = {},
      const std::vector<int>& good_feature_ids = {},
      const double timestamps[] = nullptr, const int iteration = 50);

  void UpdateLiDARAttribute(double scan_time_min, double scan_time_max);

  void Log(std::string descri) const;

  const ImuStateEstimator::Ptr GetIMUStateEstimator() const {
    return imu_state_estimator_;
  }

  void ExtendTrajectory(int64_t max_time_ns);

  IMUBias GetLatestBias() const {
    IMUBias bias;
    bias = all_imu_bias_.rbegin()->second;
    return bias;
  }

  const VPointCloud& GetMargCtrlPoint() const { return marg_ctrl_point_; }
  const VPointCloud& GetInitCtrlPoint() const { return init_ctrl_point_; }

  Eigen::Quaterniond GetGlobalFrame() const {
    Eigen::Vector3d z_axis = gravity_ / gravity_.norm();
    Eigen::Vector3d e_1(1, 0, 0);
    Eigen::Vector3d x_axis = e_1 - z_axis * z_axis.transpose() * e_1;
    x_axis = x_axis / x_axis.norm();
    Eigen::Matrix<double, 3, 1> y_axis =
        Eigen::SkewSymmetric<double>(z_axis) * x_axis;

    Eigen::Matrix<double, 3, 3> Rot;
    Rot.block<3, 1>(0, 0) = x_axis;
    Rot.block<3, 1>(0, 1) = y_axis;
    Rot.block<3, 1>(0, 2) = z_axis;

    Eigen::Matrix3d R_Map_To_G = Rot.inverse();
    Eigen::Quaterniond q_MtoG(R_Map_To_G);
    return q_MtoG;
  }

  bool verbose;

  const std::map<int, double>& GetFeatureInvDepths() const {
    return fea_id_inv_depths_;
  }

 private:
  bool LocatedInFirstSegment(double cur_t) const {
    size_t knot_idx = trajectory_->GetCtrlIndex(cur_t * S_TO_NS);
    if (knot_idx < SplineOrder)
      return true;
    else
      return false;
  }

  // get the involved imu of this optimizations
  void UpdateIMUInlio();

  void RemoveIMUData(double t_window_min);

  void InitTrajWithPropagation();

  // (R0.t0) is the pose of oldest frame before opt
  void TranfromTraj4DoF(double t_min, double t_max, const Eigen::Matrix3d& R0,
                        const Eigen::Vector3d& t0, bool apply = true);

  // opt window [t_min, t_max];
  // pose before opt (R_bef, p_bef); 
  // pose after opt (R_aft, p_aft)
  void TranfromTraj4DoF(double t_min, double t_max,
                        const Eigen::Matrix3d& R_bef,
                        const Eigen::Vector3d& p_bef,
                        const Eigen::Matrix3d& R_aft,
                        const Eigen::Vector3d& p_aft, bool apply = true);

  ///==================== Measurements
  // start pose
  PoseData original_pose_;

  // IMU raw measurememts
  Eigen::aligned_vector<IMUData> imu_data_;

  //===================== State
  TimeParam tparam_;

  OptWeight opt_weight_;

  Eigen::Vector3d gravity_;

  Trajectory::Ptr trajectory_;

  ImuStateEstimator::Ptr imu_state_estimator_;

  std::map<double, IMUBias> all_imu_bias_;

  // Marginazation info [lio system]
  MarginalizationInfo::Ptr lidar_marg_info;
  std::vector<double*> lidar_marg_parameter_blocks;
  std::pair<int, int> lidar_prior_ctrl_id;

  MarginalizationInfo::Ptr cam_marg_info;
  std::vector<double*> cam_marg_parameter_blocks;

  bool updated_loop_;

  bool opt_time_offset_;
  bool start_opt_time_offset_;

  VPointCloud marg_ctrl_point_;
  VPointCloud init_ctrl_point_;

  std::map<int, double> fea_id_inv_depths_;
};

}  // namespace clic
