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

#include <ceres/ceres.h>
#include <ceres/covariance.h>
#include <estimator/factor/ceres_local_param.h>
#include <inertial/imu_state_estimator.h>
#include <lidar_odometry/lidar_feature.h>
#include <utils/parameter_struct.h>

#include "trajectory_estimator_options.h"
#include "visual_odometry/integration_base.h"

#include <estimator/factor/analytic_diff/image_feature_factor.h>
#include <estimator/factor/analytic_diff/lidar_feature_factor.h>
#include <estimator/factor/analytic_diff/marginalization_factor.h>
#include <estimator/factor/analytic_diff/trajectory_value_factor.h>

// IMUGlobalVelocityFactor
#include <estimator/factor/auto_diff/trajectory_value_factor.h>

namespace clic {

struct ResidualSummary {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  std::map<ResidualType, std::vector<double>> err_type_sum;
  std::map<ResidualType, int> err_type_number;

  std::map<ResidualType, time_span_t> err_type_duration;
  std::map<int, std::pair<size_t, size_t>> opt_knot;

  std::string descri_info;

  ResidualSummary(std::string descri = "") : descri_info(descri) {
    for (auto typ = RType_Pose; typ <= RType_Prior;
         typ = ResidualType(typ + 1)) {
      err_type_sum[typ].push_back(0);
      err_type_number[typ] = 0;
      err_type_duration[typ] = std::make_pair(0, 0);
    }
    opt_knot[0] = std::make_pair(1, 0);  // pos knot
    opt_knot[1] = std::make_pair(1, 0);  // rot knot
  }

  void AddResidualInfo(ResidualType r_type,
                       const ceres::CostFunction* cost_function,
                       const std::vector<double*>& param_vec);

  void AddResidualTimestamp(ResidualType r_type, int64_t time_ns) {
    auto& t_span = err_type_duration[r_type];
    if (t_span.first == 0) {
      t_span.first = time_ns;
      t_span.second = time_ns;
    } else {
      t_span.first = t_span.first < time_ns ? t_span.first : time_ns;
      t_span.second = t_span.second > time_ns ? t_span.second : time_ns;
    }
  }

  void AddKnotIdx(size_t knot, bool is_pos_knot) {
    int k = is_pos_knot ? 0 : 1;
    if (opt_knot[k].first > opt_knot[k].second) {
      opt_knot[k].first = knot;
      opt_knot[k].second = knot;
    } else {
      opt_knot[k].first = opt_knot[k].first < knot ? opt_knot[k].first : knot;
      opt_knot[k].second =
          opt_knot[k].second > knot ? opt_knot[k].second : knot;
    }
  }

  void PrintSummary(int64_t t0_ns, int64_t dt_ns,
                    int fixed_ctrl_idx = -1) const;

  std::string GetTimeString(int64_t knot_min, int64_t knot_max, int64_t t0_ns,
                            int64_t dt_ns) const;

  std::string GetCtrlString(int64_t t_min_ns, int64_t t_max_ns, int64_t t0_ns,
                            int64_t dt_ns) const;
};

class TrajectoryEstimator {
  static ceres::Problem::Options DefaultProblemOptions() {
    ceres::Problem::Options options;
    options.loss_function_ownership = ceres::TAKE_OWNERSHIP;
    options.local_parameterization_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
    return options;
  }

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef std::shared_ptr<TrajectoryEstimator> Ptr;

  TrajectoryEstimator(Trajectory::Ptr trajectory,
                      TrajectoryEstimatorOptions& option,
                      std::string descri = "");

  ~TrajectoryEstimator() {
    // Ceres will call delete on local_parameterization upon completion.

    if (analytic_local_parameterization_)
      delete analytic_local_parameterization_;

    if (auto_diff_local_parameterization_)
      delete auto_diff_local_parameterization_;

    if (homo_vec_local_parameterization_)
      delete homo_vec_local_parameterization_;

    // delete marginalization_info_;
  }

  /// fixed the ctrl points before that scan
  void SetKeyScanConstant(double max_time);

  /// check if the measurement within the opt window
  bool MeasuredTimeToNs(const SensorType& sensor_type, const double& timestamp,
                        int64_t& time_ns) const;

  /// directly fix the specify ctrl point
  void SetFixedIndex(int idx) { fixed_control_point_index_ = idx; }

  int GetFixedControlIndex() const { return fixed_control_point_index_; }

  void SetTimeoffsetState();

  // [factor] start pose of the trajectory
  void AddStartTimePose(const PoseData& pose);

  // [factor] pose factor
  void AddPoseMeasurementAnalytic(const PoseData& pose_data,
                                  const Eigen::Matrix<double, 6, 1>& info_vec);
  // [factor] imu factor
  void AddIMUMeasurementAnalytic(const IMUData& imu_data, double* gyro_bias,
                                 double* accel_bias, double* gravity,
                                 const Eigen::Matrix<double, 6, 1>& info_vec,
                                 bool marg_this_factor = false);

  // [factor] bias factor
  void AddBiasFactor(double* bias_gyr_i, double* bias_gyr_j, double* bias_acc_i,
                     double* bias_acc_j, double dt,
                     const Eigen::Matrix<double, 6, 1>& info_vec,
                     bool marg_this_factor = false, bool marg_all_bias = false);

  void AddGravityFactor(double* gravity, const Eigen::Vector3d& info_vec,
                        bool marg_this_factor = false);

  // [factor] preIntegration factor ti(ta), tj(tb)
  void AddPreIntegrationAnalytic(double ti, double tj,
                                 IntegrationBase* pre_integration,
                                 double* gyro_bias_i, double* gyro_bias_j,
                                 double* accel_bias_i, double* accel_bias_j,
                                 bool marg_this_factor = false);
  // [factor] Relative Rotation factor
  void AddRelativeRotationAnalytic(double ta, double tb, const SO3d& S_BtoA,
                                   const Eigen::Vector3d& info_vec);

  // [factor] Global Velocity factor
  bool AddGlobalVelocityMeasurement(const double timestamp,
                                    const Eigen::Vector3d& global_v,
                                    double vel_weight);

  // [factor] Local Velocity factor
  void AddLocalVelocityMeasurementAnalytic(const double timestamp,
                                           const Eigen::Vector3d& local_v,
                                           double weight);

  // [factor] lidar Measurement
  void AddLoamMeasurementAnalytic(const PointCorrespondence& pc,
                                  const SO3d& S_GtoM,
                                  const Eigen::Vector3d& p_GinM,
                                  const SO3d& S_LtoI,
                                  const Eigen::Vector3d& p_LinI, double weight,
                                  bool marg_this_factor = false);

  // [factor] image reprojection factor
  void AddImageFeatureAnalytic(const double ti, const Eigen::Vector3d& pi,
                               const double tj, const Eigen::Vector3d& pj,
                               double* inv_depth, bool fixed_depth = false,
                               bool marg_this_fearure = false);

  void AddImageDepthAnalytic(const Eigen::Vector3d& p_i,
                             const Eigen::Vector3d& p_j, const SO3d& S_CitoCj,
                             const Eigen::Vector3d& p_CiinCj,
                             double* inv_depth);

  // [factor] prior factor
  void AddMarginalizationFactor(
      MarginalizationInfo::Ptr last_marginalization_info,
      std::vector<double*>& last_marginalization_parameter_blocks);

  ///======== AutoDiff
  // [factor AutoDiff] pose factor
  void AddPoseMeasurementAutoDiff(const PoseData& pose_data, double pos_weight,
                                  double rot_weight);

  // [factor AutoDiff] 6DoF local velocity
  void Add6DofLocalVelocityAutoDiff(
      const double timestamp, const Eigen::Matrix<double, 6, 1>& local_gyro_vel,
      double gyro_weight, double velocity_weight);

  void AddCallback(const std::vector<std::string>& descriptions,
                   const std::vector<size_t>& block_size,
                   std::vector<double*>& param_block);

  ceres::Solver::Summary Solve(int max_iterations = 50, bool progress = false,
                               int num_threads = -1);

  void PrepareMarginalizationInfo(ResidualType r_type,
                                  ceres::CostFunction* cost_function,
                                  ceres::LossFunction* loss_function,
                                  std::vector<double*>& parameter_blocks,
                                  std::vector<int>& drop_set);

  void SaveMarginalizationInfo(MarginalizationInfo::Ptr& marg_info_out,
                               std::vector<double*>& marg_param_blocks_out);

  const ResidualSummary& GetResidualSummary() const {
    return residual_summary_;
  }

 private:
  void AddControlPoints(const SplineMeta<SplineOrder>& spline_meta,
                        std::vector<double*>& vec, bool addPosKnot = false);

  void PrepareMarginalizationInfo(ResidualType r_type,
                                  const SplineMeta<SplineOrder>& spline_meta,
                                  ceres::CostFunction* cost_function,
                                  ceres::LossFunction* loss_function,
                                  std::vector<double*>& parameter_blocks,
                                  std::vector<int>& drop_set_wo_ctrl_point);

  bool IsParamUpdated(const double* values) const;

 public:
  TrajectoryEstimatorOptions options;

 private:
  Trajectory::Ptr trajectory_;

  std::shared_ptr<ceres::Problem> problem_;
  ceres::LocalParameterization* analytic_local_parameterization_;
  ceres::HomogeneousVectorParameterization* homo_vec_local_parameterization_;

  ceres::LocalParameterization* auto_diff_local_parameterization_;

  std::map<SensorType, double*> t_offset_ns_opt_params_;

  int fixed_control_point_index_;

  // Marginalization
  MarginalizationInfo::Ptr marginalization_info_;

  // for debug
  ResidualSummary residual_summary_;

  bool callback_needs_state_;
  std::vector<std::unique_ptr<ceres::IterationCallback>> callbacks_;
};

}  // namespace clic
