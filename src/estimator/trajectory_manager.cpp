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

#include <estimator/factor/analytic_diff/image_feature_factor.h>
#include <estimator/factor/analytic_diff/trajectory_value_factor.h>
#include <estimator/trajectory_manager.h>
#include <ros/assert.h>
#include <utils/log_utils.h>

namespace clic {

TrajectoryManager::TrajectoryManager(const YAML::Node& node,
                                     Trajectory::Ptr trajectory)
    : verbose(false),
      opt_weight_(OptWeight(node)),
      trajectory_(trajectory),
      lidar_marg_info(nullptr),
      cam_marg_info(nullptr) {
  std::string config_path = node["config_path"].as<std::string>();
  std::string imu_yaml = node["imu_yaml"].as<std::string>();
  YAML::Node imu_node = YAML::LoadFile(config_path + imu_yaml);
  imu_state_estimator_ = std::make_shared<ImuStateEstimator>(imu_node);

  opt_time_offset_ = yaml::GetBool(node, "opt_time_offset");
  std::cout << YELLOW << "\t time offset calibration ? "
            << (opt_time_offset_ ? "yes" : "no") << "\n"
            << RESET;
  start_opt_time_offset_ = false;

  lidar_prior_ctrl_id = std::make_pair(0, 0);

  InitFactorInfo(trajectory_->GetSensorEP(CameraSensor),
                 trajectory_->GetSensorEP(LiDARSensor),
                 opt_weight_.image_weight);
}

void TrajectoryManager::InitFactorInfo(
    const ExtrinsicParam& Ep_CtoI, const ExtrinsicParam& Ep_LtoI,
    const double image_feature_weight) {
  if (image_feature_weight > 1e-5) {
    Eigen::Matrix2d sqrt_info =
        image_feature_weight * Eigen::Matrix2d::Identity();

    analytic_derivative::ImageFeatureFactor::SetParam(Ep_CtoI.so3, Ep_CtoI.p);
    analytic_derivative::ImageFeatureFactor::sqrt_info = sqrt_info;

    analytic_derivative::Image3D2DFactor::SetParam(Ep_CtoI.so3, Ep_CtoI.p);
    analytic_derivative::Image3D2DFactor::sqrt_info = sqrt_info;

    analytic_derivative::ImageFeatureOnePoseFactor::SetParam(Ep_CtoI.so3,
                                                             Ep_CtoI.p);
    analytic_derivative::ImageFeatureOnePoseFactor::sqrt_info = sqrt_info;

    analytic_derivative::ImageDepthFactor::sqrt_info = sqrt_info;

    analytic_derivative::EpipolarFactor::SetParam(Ep_CtoI.so3, Ep_CtoI.p);
  }
  analytic_derivative::LoamFeatureOptMapPoseFactor::SetParam(Ep_LtoI.so3,
                                                             Ep_LtoI.p);
  analytic_derivative::RalativeLoamFeatureFactor::SetParam(Ep_LtoI.so3,
                                                           Ep_LtoI.p);
}

void TrajectoryManager::SetSystemState(const SystemState& sys_state) {
  gravity_ = sys_state.g;
  double t = std::min(0.0, trajectory_->minTime(LiDARSensor));
  all_imu_bias_[t] = sys_state.bias;

  // TODO
  all_imu_bias_[t].gyro_bias = Eigen::Vector3d::Zero();
  all_imu_bias_[t].accel_bias = Eigen::Vector3d::Zero();

  SetOriginalPose(sys_state.q, sys_state.p);

  SO3d R0(sys_state.q);
  for (size_t i = 0; i <= trajectory_->numKnots(); i++) {
    trajectory_->setKnotSO3(R0, i);
  }

  Eigen::Vector3d euler = R2ypr(sys_state.q.toRotationMatrix());
  std::cout << "SystemState:\n";
  std::cout << "\t- position: " << sys_state.p.transpose() << std::endl;
  std::cout << "\t- euler: " << euler.transpose() << std::endl;
  std::cout << "\t- gravity: " << gravity_.transpose() << std::endl;
  std::cout << "\t- gyr bias: " << sys_state.bias.gyro_bias.transpose()
            << std::endl;
  std::cout << "\t- acc bias: " << sys_state.bias.accel_bias.transpose()
            << std::endl;
}

void TrajectoryManager::SetOriginalPose(Eigen::Quaterniond q,
                                        Eigen::Vector3d p) {
  original_pose_.orientation.setQuaternion(q);
  original_pose_.position = p;
}

void TrajectoryManager::AddIMUData(const IMUData& data) {
  if (trajectory_->GetDataStartTime() < 0) {
    trajectory_->SetDataStartTime(data.timestamp);
  }
  imu_data_.emplace_back(data);
  imu_data_.back().timestamp -= trajectory_->GetDataStartTime();
  imu_state_estimator_->FeedIMUData(imu_data_.back());
}

void TrajectoryManager::RemoveIMUData(double t_window_min) {
  if (t_window_min < trajectory_->minTime(IMUSensor)) return;
  int knot_index =
      trajectory_->GetCtrlIndex(t_window_min * S_TO_NS) + 1 - SplineOrder;
  if (knot_index <= 0) return;
  double active_time =
      knot_index * trajectory_->getDt() + trajectory_->minTime(IMUSensor);

  // https://stackoverflow.com/questions/991335/
  // how-to-erase-delete-pointers-to-objects-stored-in-a-vector
  for (auto iter = imu_data_.begin(); iter != imu_data_.end();) {
    if (iter->timestamp < active_time) {
      iter = imu_data_.erase(iter);
    } else {
      break;
    }
  }
}

void TrajectoryManager::UpdateIMUInlio() {
  int idx = trajectory_->GetCtrlIndex(tparam_.cur_scan[0] * S_TO_NS);
  // idx -= 1;

  double t_min = trajectory_->minTime(LiDARSensor) + trajectory_->getDt() * idx;
  if (t_min < 0) t_min = 0;
  // double t_min = std::min(tparam_.last_scan[1], tparam_.cur_scan[0]);
  double t_max = trajectory_->maxTime(LiDARSensor);

  double t_offset_imu =
      trajectory_->GetSensorEP(IMUSensor).t_offset_ns * NS_TO_S;

  if (imu_data_.front().timestamp > trajectory_->maxTime(IMUSensor)) {
    std::cout << RED << "[UpdateIMUInlio] imu in "
              << imu_data_.front().timestamp << ", "
              << imu_data_.back().timestamp << "; t offset " << t_offset_imu
              << "; traj max " << t_max << "\n"
              << RESET;

    std::cout << "[UpdateIMUInlio] lidar in " << tparam_.cur_scan[0] << ", "
              << tparam_.cur_scan[1] << "\n";
  }

  for (auto iter = imu_data_.begin(); iter != imu_data_.end(); ++iter) {
    if (iter->timestamp + t_offset_imu >= t_min) {
      if (iter->timestamp + t_offset_imu >= t_max) {
        std::cout << RED << "[UpdateIMUInlio] imu at " << iter->timestamp
                  << "; t offset " << t_offset_imu << "; traj max " << t_max
                  << "; distance: " << std::distance(imu_data_.begin(), iter)
                  << "\n"
                  << RESET;
        continue;
      }
      tparam_.lio_imu_idx[0] = std::distance(imu_data_.begin(), iter);
      tparam_.lio_imu_time[0] = iter->timestamp + t_offset_imu;
      break;
    }
  }

  for (auto rter = imu_data_.rbegin(); rter != imu_data_.rend(); ++rter) {
    if (rter->timestamp + t_offset_imu < t_max) {
      tparam_.lio_imu_idx[1] =
          std::distance(imu_data_.begin(), rter.base()) - 1;
      tparam_.lio_imu_time[1] = rter->timestamp + t_offset_imu;
      break;
    }
  }
}

void TrajectoryManager::ExtendTrajectory(int64_t max_time_ns) {
  SE3d last_knot = trajectory_->getLastKnot();
  trajectory_->extendKnotsTo(max_time_ns, last_knot);
}

void TrajectoryManager::PropagateTrajectory(double scan_time_min,
                                            double scan_time_max) {
  if (imu_data_.empty() || imu_data_.size() == 1) {
    LOG(ERROR) << "[AppendWithIMUData] IMU data empty! ";
    return;
  }

  ExtendTrajectory(scan_time_max * S_TO_NS);

  IMUState imu_state;
  double inte_start_time = 0.0;

  static bool first_inte_time = true;
  if (first_inte_time) {
    first_inte_time = false;
    imu_state.timestamp = inte_start_time;
    imu_state.q = original_pose_.orientation.unit_quaternion();
    imu_state.p = original_pose_.position;
    imu_state.v = Eigen::Vector3d(0, 0, 0);
  } else {
    double t_imu = trajectory_->GetForcedFixedTime() -
                   trajectory_->GetSensorEP(IMUSensor).t_offset_ns * NS_TO_S;

    inte_start_time = t_imu > 0 ? t_imu : 0;
    imu_state.timestamp = t_imu;

    trajectory_->GetIMUState(inte_start_time, imu_state);
  }
  imu_state.bias = all_imu_bias_.rbegin()->second;
  imu_state.g = gravity_;
  imu_state_estimator_->Propagate(imu_state, inte_start_time,
                                  trajectory_->maxTime(IMUSensor));
  // the involved imu measurements during the optimization
  tparam_.UpdateCurScan(scan_time_min, scan_time_max);
  UpdateIMUInlio();

  InitTrajWithPropagation();
}

void TrajectoryManager::InitTrajWithPropagation() {
  LOG(INFO) << "[InitTrajWithPropagation] imu data time = ["
            << tparam_.lio_imu_time[0] << ", " << tparam_.lio_imu_time[1]
            << "]";
  LOG(INFO) << " traj active time " << trajectory_->GetActiveTime()
            << " traj max time = " << trajectory_->maxTime(LiDARSensor);

  TrajectoryEstimatorOptions option;
  option.lock_ab = true;
  option.lock_wb = true;
  option.lock_g = true;
  option.show_residual_summary = verbose;
  TrajectoryEstimator::Ptr estimator(
      new TrajectoryEstimator(trajectory_, option, "Init Traj"));

  double opt_min_time = tparam_.lio_imu_time[0];
  // double opt_max_time = trajectory_->maxTime(LiDARSensor);

  LOG(INFO) << "[InitTrajWithPropagation] traj involved: ["
            << trajectory_->GetCtrlIndex(opt_min_time * S_TO_NS) << ","
            << trajectory_->numKnots() << "]";

  // SE3d pose0 = trajectory_->GetIMUPose(opt_min_time);

  if (LocatedInFirstSegment(opt_min_time)) {
    estimator->SetFixedIndex(trajectory_->N - 1);
    // estimator->AddStartTimePose(original_pose_);
  } else {
    estimator->SetFixedIndex(lidar_prior_ctrl_id.first - 1);
    // estimator->SetKeyScanConstant(trajectory_->GetForcedFixedTime());
  }

  trajectory_->opt_min_init_time_tmp = opt_min_time;
  trajectory_->opt_init_fixed_idx_tmp = estimator->GetFixedControlIndex();

  // [0] prior
  if (true && lidar_marg_info) {
    estimator->AddMarginalizationFactor(lidar_marg_info,
                                        lidar_marg_parameter_blocks);
  }

  // [1] IMU 
  double* para_bg = all_imu_bias_.rbegin()->second.gyro_bias.data();
  double* para_ba = all_imu_bias_.rbegin()->second.accel_bias.data();
  for (int i = tparam_.lio_imu_idx[0]; i <= tparam_.lio_imu_idx[1]; ++i) {
    estimator->AddIMUMeasurementAnalytic(imu_data_.at(i), para_bg, para_ba,
                                         gravity_.data(),
                                         opt_weight_.imu_info_vec);
  }
  // [2] IMU velocity
  // if (imu_odom.timestamp >= opt_min_time &&
  //     imu_odom.timestamp <= opt_max_time)
  {
    const IMUState& imu_odom = imu_state_estimator_->GetIMUState();
    bool ret = estimator->AddGlobalVelocityMeasurement(
        imu_odom.timestamp, imu_odom.v, opt_weight_.global_velocity);
    if (ret) {
      LOG(INFO) << "[InitTrajWithPropagation] AddGlobalVelocityMeasurement";
    }
  }

  estimator->SetTimeoffsetState();
  ceres::Solver::Summary summary = estimator->Solve(50, false);
  LOG(INFO) << summary.BriefReport();
  LOG(INFO) << "TrajInit Successful/Unsuccessful steps: "
            << summary.num_successful_steps << "/"
            << summary.num_unsuccessful_steps;

  size_t opt_start_idx = trajectory_->GetCtrlIndex(opt_min_time * S_TO_NS);
  init_ctrl_point_.clear();
  for (size_t i = opt_start_idx; i < trajectory_->numKnots(); ++i) {
    const auto& p = trajectory_->getKnotPos(i);
    VPoint vp;
    vp.x = p[0];
    vp.y = p[1];
    vp.z = p[2];
    init_ctrl_point_.push_back(vp);
  }

  // if we don't have prior，rigidly transfrom the traj
  // if (lidar_marg_info == nullptr)
  // TranfromTraj4DoF(opt_min_time, opt_max_time, pose0.so3().matrix(),
  //                  pose0.translation());
}

void TrajectoryManager::UpdateLIOPrior(
    const Eigen::aligned_vector<PointCorrespondence>& point_corrs) {
  // factor: IMUfactor, lidar factor
  // marg : [old ctrl point], last_bias, gravity
  // prior: ctrl point, cur_bias

  TrajectoryEstimatorOptions option;
  option.is_marg_state = true;
  option.marg_bias_param = false;  
  option.marg_gravity_param = true;
  if (opt_time_offset_ && start_opt_time_offset_) {
    option.marg_t_offset_param = false;
  } else {
    option.marg_t_offset_param = true;
  }

  // decide the ctrl points to be margalized
  double opt_min_time = std::min(tparam_.cur_scan[0], tparam_.lio_imu_time[0]);
  int opt_idx = trajectory_->GetCtrlIndex(opt_min_time * S_TO_NS);
  option.ctrl_to_be_opt_now = std::min(lidar_prior_ctrl_id.first, opt_idx);

  int scan_idx = trajectory_->GetCtrlIndex(tparam_.cur_scan[1] * S_TO_NS);
  option.ctrl_to_be_opt_later = std::max(scan_idx, trajectory_->N);

  auto& cur_bias = all_imu_bias_[tparam_.cur_scan[0]];
  auto& last_bias = all_imu_bias_[tparam_.last_scan[0]];

  TrajectoryEstimator::Ptr estimator(
      new TrajectoryEstimator(trajectory_, option));

  /// [step0] add marginalization_factor
  if (lidar_marg_info) {
    std::vector<double*> drop_param_set;
    for (int i = option.ctrl_to_be_opt_now; i < option.ctrl_to_be_opt_later;
         ++i) {
      drop_param_set.emplace_back(trajectory_->getKnotSO3(i).data());
      drop_param_set.emplace_back(trajectory_->getKnotPos(i).data());
    }
    // last bias
    drop_param_set.emplace_back(last_bias.accel_bias.data());  // in the prior
    drop_param_set.emplace_back(last_bias.gyro_bias.data());

    std::vector<int> drop_set;
    for (int j = 0; j < (int)lidar_marg_parameter_blocks.size(); j++) {
      for (auto const& drop_param : drop_param_set) {
        if (lidar_marg_parameter_blocks[j] == drop_param) {
          drop_set.emplace_back(j);
          break;
        }
      }
    }
    if (!drop_set.empty()) {
      MarginalizationFactor* marginalization_factor =
          new MarginalizationFactor(lidar_marg_info);

      estimator->PrepareMarginalizationInfo(RType_Prior, marginalization_factor,
                                            NULL, lidar_marg_parameter_blocks,
                                            drop_set);
    }
  }

/// [step1] add lidar features
#if true
  bool marg_lidar_factor = true;
  SO3d S_LtoI = trajectory_->GetSensorEP(LiDARSensor).so3;
  Eigen::Vector3d p_LinI = trajectory_->GetSensorEP(LiDARSensor).p;
  SE3d T_MtoG = trajectory_->GetLidarPose(point_corrs.front().t_map);
  SO3d S_GtoM = T_MtoG.so3().inverse();
  Eigen::Vector3d p_GinM = S_GtoM * (-T_MtoG.translation());

  bool in_target_frame = false;
  if (!in_target_frame) {
    S_GtoM = SO3d(Eigen::Quaterniond::Identity());
    p_GinM = Eigen::Vector3d::Zero();
  }

  for (const auto& v : point_corrs) {
    estimator->AddLoamMeasurementAnalytic(v, S_GtoM, p_GinM, S_LtoI, p_LinI,
                                          opt_weight_.lidar_weight,
                                          marg_lidar_factor);
  }
#endif

#if true
  int idx = trajectory_->GetCtrlIndex(tparam_.cur_scan[1] * S_TO_NS);
  double t_cur_scan_low_bound =
      trajectory_->minTime(LiDARSensor) + trajectory_->getDt() * idx;
  if (t_cur_scan_low_bound < tparam_.lio_imu_time[0])
    t_cur_scan_low_bound = tparam_.lio_imu_time[0];

  double t_offset_imu =
      trajectory_->GetSensorEP(IMUSensor).t_offset_ns * NS_TO_S;
  double prior_min_time = trajectory_->maxTime(LiDARSensor);
  double prior_max_time = -1;
  // [2] IMU 
  bool marg_imu_factor = true;
  for (int i = tparam_.lio_imu_idx[0]; i < tparam_.lio_imu_idx[1]; ++i) {
    double t_imu = imu_data_.at(i).timestamp + t_offset_imu;

    if (t_imu > t_cur_scan_low_bound) break;

    if (t_imu > tparam_.cur_scan[0]) {
      option.marg_bias_param = false;
      estimator->AddIMUMeasurementAnalytic(
          imu_data_.at(i), cur_bias.gyro_bias.data(),
          cur_bias.accel_bias.data(), gravity_.data(), opt_weight_.imu_info_vec,
          marg_imu_factor);
    } else {
      option.marg_bias_param = true;
      estimator->AddIMUMeasurementAnalytic(
          imu_data_.at(i), last_bias.gyro_bias.data(),
          last_bias.accel_bias.data(), gravity_.data(),
          opt_weight_.imu_info_vec, marg_imu_factor);
    }

    prior_min_time = prior_min_time < t_imu ? prior_min_time : t_imu;
    prior_max_time = prior_max_time > t_imu ? prior_max_time : t_imu;
  }
  LOG(INFO) << "[LIO Prior] IMU data [" << prior_min_time << ","
            << prior_max_time << "]";
#endif

  estimator->SaveMarginalizationInfo(lidar_marg_info,
                                     lidar_marg_parameter_blocks);

  // ======= debug output ======= //
  if (lidar_marg_info) {
    lidar_prior_ctrl_id.first = option.ctrl_to_be_opt_later;
    lidar_prior_ctrl_id.second = trajectory_->numKnots() - 1;

    LOG(INFO) << "[After LIO Prior]  marg/left: " << lidar_marg_info->m << "/"
              << lidar_marg_info->n;
    LOG(INFO) << "[LIO Prior Ctrl ID] = [" << lidar_prior_ctrl_id.first << ","
              << lidar_prior_ctrl_id.second << "] "
              << lidar_prior_ctrl_id.second - lidar_prior_ctrl_id.first;
  }

  marg_ctrl_point_.clear();
  for (int i = lidar_prior_ctrl_id.first; i < lidar_prior_ctrl_id.second; ++i) {
    const auto& p = trajectory_->getKnotPos(i);
    VPoint vp;
    vp.x = p[0];
    vp.y = p[1];
    vp.z = p[2];
    marg_ctrl_point_.push_back(vp);
  }
}

void TrajectoryManager::UpdateVisualOffsetPrior(
    const std::list<FeaturePerId>& features,
    const std::vector<int>& good_feature_ids, const double timestamps[]) {
  if (!(opt_time_offset_ && start_opt_time_offset_)) {
    return;
  }

  // factor: image factor( oldest frame )
  // marg :  ctrl point, inv_depth
  // prior:  t_offset_camera
  TrajectoryEstimatorOptions option;
  option.is_marg_state = true;
  option.marg_t_offset_param = false;
  option.lock_EPs.at(CameraSensor).lock_t_offset = true;

  option.ctrl_to_be_opt_now = 0;  // idx
  option.ctrl_to_be_opt_later = lidar_prior_ctrl_id.first;

  TrajectoryEstimator::Ptr estimator(
      new TrajectoryEstimator(trajectory_, option));

  if (lidar_marg_info) {
    std::vector<int> drop_set;
    MarginalizationFactor* marginalization_factor =
        new MarginalizationFactor(lidar_marg_info);
    estimator->PrepareMarginalizationInfo(RType_Prior, marginalization_factor,
                                          NULL, lidar_marg_parameter_blocks,
                                          drop_set);
  }

  int idx_marg = WINDOW_SIZE;
  double t_offset_ns = trajectory_->GetSensorEP(CameraSensor).t_offset_ns;
  for (; idx_marg >= 0; idx_marg--) {
    int64_t t_cam = timestamps[idx_marg] * S_TO_NS + t_offset_ns;
    int ctrl_idx = trajectory_->GetCtrlIndex(t_cam);
    if (ctrl_idx < lidar_prior_ctrl_id.first) {
      break;
    }
  }

  if (idx_marg <= 0) return;

  int cnt_fea = 0, cnt_factor = 0;
  for (const auto& it_per_id : features) {
    if (std::find(good_feature_ids.begin(), good_feature_ids.end(),
                  it_per_id.feature_id) == good_feature_ids.end())
      continue;
    if (it_per_id.start_frame != 0) continue;
    if (it_per_id.endFrame() != WINDOW_SIZE) continue;

    int idx_i = it_per_id.start_frame;
    int idx_j = it_per_id.start_frame - 1;

    double ti = timestamps[idx_i];
    Eigen::Vector3d pts_i = it_per_id.feature_per_frame[0].point;

    bool fixed_depth = true;
    bool marg_this_fearure = true;
    for (auto& it_per_frame : it_per_id.feature_per_frame) {
      if (++idx_j == idx_i) continue;
      // if (idx_j > WINDOW_SIZE - 4) break;

      if (idx_j < idx_marg) continue;

      double tj = timestamps[idx_j];
      Eigen::Vector3d pts_j = it_per_frame.point;

      estimator->AddImageFeatureAnalytic(
          ti, pts_i, tj, pts_j, &fea_id_inv_depths_[it_per_id.feature_id],
          fixed_depth, marg_this_fearure);
      cnt_factor++;
    }
    cnt_fea++;
    if (cnt_fea > 50) break;
  }

  LOG(INFO) << "AddImageFeatureAnalytic " << cnt_fea;

  if (cnt_factor > 0) {
    estimator->SaveMarginalizationInfo(lidar_marg_info,
                                       lidar_marg_parameter_blocks);
    LOG(INFO) << "[UpdateVisualOffsetPrior]  marg/left: " << lidar_marg_info->m
              << "/" << lidar_marg_info->n;
  } else {
    LOG(INFO) << "[UpdateVisualOffsetPrior] empty valid landmark.";
  }
}

bool TrajectoryManager::UpdateTrajectoryWithLoamFeature(
    const Eigen::aligned_vector<PointCorrespondence>& point_corrs,
    const std::list<FeaturePerId>& features,
    const std::vector<int>& good_feature_ids, const double timestamps[],
    const int iteration) {
  if (point_corrs.empty() || imu_data_.empty()) {
    LOG(WARNING) << " input empty data " << point_corrs.size() << ", "
                 << imu_data_.size();
    return false;
  }

  LOG(INFO) << "[UpdateTrajWithLoamFeature] lidar_point_corrs: "
            << point_corrs.size() << "; t_map: " << point_corrs.front().t_map
            << "; scan = [" << tparam_.cur_scan[0] << ", "
            << tparam_.cur_scan[1] << "]; scan_cor = ["
            << point_corrs.front().t_point << ", " << point_corrs.back().t_point
            << "]";

  LOG(INFO) << "[UpdateTrajWithLoamFeature] imu_idx/size: "
            << tparam_.lio_imu_idx[0] << "/" << imu_data_.size() << "; imu = ["
            << tparam_.lio_imu_time[0] << ", " << tparam_.lio_imu_time[1]
            << "]";

  bool has_image = false;
  if (!good_feature_ids.empty() && timestamps != nullptr) {
    has_image = true;
  }

  if (tparam_.last_scan[0] < 0) {
    tparam_.last_scan[0] = 0;
    IMUBias last_bias = all_imu_bias_.rbegin()->second;
    all_imu_bias_[tparam_.last_scan[0]] = last_bias;
  }
  // add new bias
  if (all_imu_bias_.rbegin()->first < tparam_.cur_scan[0]) {
    IMUBias last_bias = all_imu_bias_.rbegin()->second;
    all_imu_bias_[tparam_.cur_scan[0]] = last_bias;
  }

  // get two latest bias
  std::vector<double> bias_time_vec;
  std::map<int, double*> para_bg_vec;
  std::map<int, double*> para_ba_vec;
  {
    auto& bias0 = all_imu_bias_[tparam_.last_scan[0]];
    para_bg_vec[0] = bias0.gyro_bias.data();
    para_ba_vec[0] = bias0.accel_bias.data();

    auto& bias1 = all_imu_bias_[tparam_.cur_scan[0]];
    para_bg_vec[1] = bias1.gyro_bias.data();
    para_ba_vec[1] = bias1.accel_bias.data();

    bias_time_vec.push_back(tparam_.last_scan[0]);
    bias_time_vec.push_back(tparam_.cur_scan[0]);
  }

  /// set the params to be optimized
  TrajectoryEstimatorOptions option;
  option.lock_ab = false;
  option.lock_wb = false;
  option.lock_g = true;
  option.show_residual_summary = verbose;
  if (opt_time_offset_ && trajectory_->maxTime(LiDARSensor) > 5 &&
      (has_image ? (timestamps[0] > 5) : true)) {
    start_opt_time_offset_ = true;
    option.lock_EPs.at(IMUSensor).lock_t_offset = false;
    option.t_offset_padding_ns = 10e6;  // 10ms
    if (has_image) {
      option.lock_EPs.at(CameraSensor).lock_t_offset = false;
      double dt_ns =
          (trajectory_->maxTime(CameraSensor) - timestamps[WINDOW_SIZE]) *
          S_TO_NS;
      if (dt_ns <= 0) {
        option.t_offset_padding_ns = 0;
      } else if (dt_ns < option.t_offset_padding_ns) {
        option.t_offset_padding_ns = dt_ns;
        // std::cout << "t_offset_padding [ms]: " << dt_ns * 1e-6 << "\n";
      }
    }
    LOG(INFO) << "t_offset_padding [ms]: " << option.t_offset_padding_ns * 1e-6;
  }

  TrajectoryEstimator::Ptr estimator(
      new TrajectoryEstimator(trajectory_, option, "Before LIO"));

  double opt_min_time = std::min(tparam_.cur_scan[0], tparam_.lio_imu_time[0]);
  // double opt_max_time = trajectory_->maxTime(LiDARSensor);

  LOG(INFO) << "[UpdateTrajWithLoamFeature] traj involved: ["
            << trajectory_->GetCtrlIndex(opt_min_time * NS_TO_S) << ","
            << trajectory_->numKnots() << "]";

  // SE3d pose0 = trajectory_->GetIMUPose(opt_min_time);

  if (LocatedInFirstSegment(opt_min_time)) {
    estimator->SetFixedIndex(trajectory_->N - 1);
    // estimator->AddStartTimePose(original_pose_);
  } else {
    estimator->SetFixedIndex(lidar_prior_ctrl_id.first - 1);
  }

  trajectory_->opt_min_lio_time_tmp = opt_min_time;

  /// [0] add lidar_marg_info factor
  if (lidar_marg_info) {
    estimator->AddMarginalizationFactor(lidar_marg_info,
                                        lidar_marg_parameter_blocks);
  }
  if (cam_marg_info) {
    estimator->AddMarginalizationFactor(cam_marg_info,
                                        cam_marg_parameter_blocks);
  }

  // [1] LiDAR
  {
    SO3d S_LtoI = trajectory_->GetSensorEP(LiDARSensor).so3;
    Eigen::Vector3d p_LinI = trajectory_->GetSensorEP(LiDARSensor).p;
    SE3d T_MtoG = trajectory_->GetLidarPose(point_corrs.front().t_map);
    SO3d S_GtoM = T_MtoG.so3().inverse();
    Eigen::Vector3d p_GinM = S_GtoM * (-T_MtoG.translation());

    bool in_target_frame = false;
    if (!in_target_frame) {
      S_GtoM = SO3d(Eigen::Quaterniond::Identity());
      p_GinM = Eigen::Vector3d::Zero();
    }

    for (const auto& v : point_corrs) {
      if (v.t_point < tparam_.last_scan[1]) continue;
      estimator->AddLoamMeasurementAnalytic(v, S_GtoM, p_GinM, S_LtoI, p_LinI,
                                            opt_weight_.lidar_weight);
    }
  }

  // [2] image
  if (has_image) {
    fea_id_inv_depths_.clear();
    int num1 = 0;
    for (const auto& it_per_id : features) {
      if (std::find(good_feature_ids.begin(), good_feature_ids.end(),
                    it_per_id.feature_id) == good_feature_ids.end())
        continue;
      if (it_per_id.endFrame() != WINDOW_SIZE) continue;

      int idx_i = it_per_id.start_frame;
      int idx_j = it_per_id.start_frame - 1;

      double ti = timestamps[idx_i];
      Eigen::Vector3d pts_i = it_per_id.feature_per_frame[0].point;
      auto pose_CitoG = trajectory_->GetCameraPose(ti);

      Eigen::Vector3d p_in_Ci = pose_CitoG.inverse() * it_per_id.pinG;
      fea_id_inv_depths_[it_per_id.feature_id] = 1 / p_in_Ci.z();

      bool fixed_depth = true;
      for (auto& it_per_frame : it_per_id.feature_per_frame) {
        if (++idx_j == idx_i) continue;

        double tj = timestamps[idx_j];
        Eigen::Vector3d pts_j = it_per_frame.point;

        estimator->AddImageFeatureAnalytic(
            ti, pts_i, tj, pts_j, &fea_id_inv_depths_[it_per_id.feature_id],
            fixed_depth);
        num1++;
      }
    }
    LOG(INFO) << "[AddImageFeatureAnalytic] num: " << num1
              << "; good_feature_ids: " << good_feature_ids.size()
              << "; window time [" << timestamps[0] << ", "
              << timestamps[WINDOW_SIZE] << "]";
  } else {
    LOG(INFO) << "[ImageFeatureOnePose] none";
  }

  // [3] IMU 
  for (int i = tparam_.lio_imu_idx[0]; i < tparam_.lio_imu_idx[1]; ++i) {
    estimator->AddIMUMeasurementAnalytic(imu_data_.at(i), para_bg_vec[1],
                                         para_ba_vec[1], gravity_.data(),
                                         opt_weight_.imu_info_vec);
  }

  // [4] bias
  if (true) {
    double delta_time = tparam_.cur_scan[0] - tparam_.last_scan[0];

    double dt = 1. / opt_weight_.imu_noise.imu_frequency;
    double cov = delta_time / dt * (dt * dt);
    Eigen::Matrix<double, 6, 1> sqrt_info =
        (1. / std::sqrt(cov)) * opt_weight_.bias_info_vec;

    estimator->AddBiasFactor(para_bg_vec[0], para_bg_vec[1], para_ba_vec[0],
                             para_ba_vec[1], 1, sqrt_info);
    LOG(INFO) << "[Bias factor] time " << tparam_.last_scan[0] << ", "
              << tparam_.cur_scan[0] << "; dt " << delta_time * 1000 << " ms";
  }

  ceres::Solver::Summary summary = estimator->Solve(iteration, false);
  LOG(INFO) << summary.BriefReport();
  LOG(INFO) << "UpdateLio Successful/Unsuccessful steps: "
            << summary.num_successful_steps << "/"
            << summary.num_unsuccessful_steps;

  LOG(INFO) << "IMU time offset at " << tparam_.cur_scan[0] << ": "
            << trajectory_->GetSensorEP(IMUSensor).t_offset_ns * NS_TO_S;
  LOG(INFO) << "Camera time offset at " << tparam_.cur_scan[0] << ": "
            << trajectory_->GetSensorEP(CameraSensor).t_offset_ns * NS_TO_S;

  // LOG(INFO) << "IMU time offset at " << tparam_.cur_scan[0] << ": "
  //           << trajectory_->GetSensorEP(IMUSensor).t_offset_ns * NS_TO_S;
  // if (has_image) {
  //   LOG(INFO) << "Camera time offset at " << tparam_.cur_scan[0] << ": "
  //             << trajectory_->GetSensorEP(CameraSensor).t_offset_ns *
  //             NS_TO_S;
  // }

  // TranfromTraj4DoF(opt_min_time, opt_max_time, pose0.so3().matrix(),
  //                  pose0.translation(), false);

  if (false) {
    Eigen::Matrix3d nTn = Eigen::Matrix3d::Zero();
    for (const auto& v : point_corrs) {
      if (GeometryType::Plane == v.geo_type) {
        nTn += v.geo_plane.head(3) * v.geo_plane.head(3).transpose();
      }
    }
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(
        nTn, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Vector3d singular_vec = svd.singularValues();
    LOG(INFO) << "[LiDAR degenerate judgement] at " << tparam_.cur_scan[0]
              << "; nTn: " << VectorToString(singular_vec) << "; "
              << VectorToString(singular_vec / singular_vec[0]);
  }

  // debug: show residual after optimization
  if (verbose) {
    TrajectoryEstimatorOptions option;
    option.lock_ab = false;
    option.lock_wb = false;
    option.show_residual_summary = verbose;
    TrajectoryEstimator::Ptr estimator(
        new TrajectoryEstimator(trajectory_, option, "After LIO"));
    /// [0] add lidar_marg_info factor
    if (lidar_marg_info) {
      estimator->AddMarginalizationFactor(lidar_marg_info,
                                          lidar_marg_parameter_blocks);
    }
    // [1] lidar
    {
      SO3d S_LtoI = trajectory_->GetSensorEP(LiDARSensor).so3;
      Eigen::Vector3d p_LinI = trajectory_->GetSensorEP(LiDARSensor).p;
      SE3d T_MtoG = trajectory_->GetLidarPose(point_corrs.front().t_map);
      SO3d S_GtoM = T_MtoG.so3().inverse();
      Eigen::Vector3d p_GinM = S_GtoM * (-T_MtoG.translation());

      bool in_target_frame = false;
      if (!in_target_frame) {
        S_GtoM = SO3d(Eigen::Quaterniond::Identity());
        p_GinM = Eigen::Vector3d::Zero();
      }

      for (const auto& v : point_corrs) {
        if (v.t_point < tparam_.last_scan[1]) continue;
        estimator->AddLoamMeasurementAnalytic(v, S_GtoM, p_GinM, S_LtoI, p_LinI,
                                              opt_weight_.lidar_weight);
      }
    }

    // [2] image
    if (has_image) {
      for (const auto& it_per_id : features) {
        if (std::find(good_feature_ids.begin(), good_feature_ids.end(),
                      it_per_id.feature_id) == good_feature_ids.end())
          continue;
        if (it_per_id.endFrame() != WINDOW_SIZE) continue;

        int idx_i = it_per_id.start_frame;
        int idx_j = it_per_id.start_frame - 1;

        double ti = timestamps[idx_i];
        Eigen::Vector3d pts_i = it_per_id.feature_per_frame[0].point;

        bool fixed_depth = true;
        for (auto& it_per_frame : it_per_id.feature_per_frame) {
          if (++idx_j == idx_i) continue;

          double tj = timestamps[idx_j];
          Eigen::Vector3d pts_j = it_per_frame.point;

          estimator->AddImageFeatureAnalytic(
              ti, pts_i, tj, pts_j, &fea_id_inv_depths_[it_per_id.feature_id],
              fixed_depth);
        }
      }
    }

    // [3] IMU
    for (int i = tparam_.lio_imu_idx[0]; i < tparam_.lio_imu_idx[1]; ++i) {
      estimator->AddIMUMeasurementAnalytic(imu_data_.at(i), para_bg_vec[1],
                                           para_ba_vec[1], gravity_.data(),
                                           opt_weight_.imu_info_vec);
    }

    // [4] bias
    if (true) {
      double delta_time = tparam_.cur_scan[0] - tparam_.last_scan[0];
      double dt = 1. / opt_weight_.imu_noise.imu_frequency;
      double cov = delta_time / dt * (dt * dt);
      Eigen::Matrix<double, 6, 1> sqrt_info =
          (1. / std::sqrt(cov)) * opt_weight_.bias_info_vec;

      estimator->AddBiasFactor(para_bg_vec[0], para_bg_vec[1], para_ba_vec[0],
                               para_ba_vec[1], 1, sqrt_info);
    }
    estimator->GetResidualSummary().PrintSummary(trajectory_->minTimeNs(),
                                                 trajectory_->getDtNs());

    LOG(INFO) << "[LIO Prior Ctrl ID] = [" << lidar_prior_ctrl_id.first << ","
              << lidar_prior_ctrl_id.second << "] "
              << lidar_prior_ctrl_id.second - lidar_prior_ctrl_id.first;
  }
  return true;
}

void TrajectoryManager::TranfromTraj4DoF(double t_min, double t_max,
                                         const Eigen::Matrix3d& R0,
                                         const Eigen::Vector3d& t0,
                                         bool apply) {
  // the pose of first frame in the window
  SE3d pose00 = trajectory_->GetIMUPose(t_min);
  Eigen::Matrix3d R00 = pose00.so3().matrix();
  Eigen::Vector3d t00 = pose00.translation();

  TranfromTraj4DoF(t_min, t_max, R0, t0, R00, t00, apply);
}

void TrajectoryManager::TranfromTraj4DoF(double t_min, double t_max,
                                         const Eigen::Matrix3d& R_bef,
                                         const Eigen::Vector3d& p_bef,
                                         const Eigen::Matrix3d& R_aft,
                                         const Eigen::Vector3d& p_aft,
                                         bool apply) {
  Eigen::Vector3d euler_bef = R2ypr(R_bef);
  Eigen::Vector3d euler_aft = R2ypr(R_aft);
  double y_diff = euler_bef.x() - euler_aft.x();


  Eigen::Matrix3d rot_diff;
  Eigen::Vector3d tran_diff;
  rot_diff = ypr2R(Eigen::Vector3d(y_diff, 0, 0));
  if (abs(abs(euler_bef.y()) - 90) < 1.0 ||
      abs(abs(euler_aft.y()) - 90) < 1.0) {
    std::cout << RED << "euler singular point!" << RESET << std::endl;
    rot_diff = R_bef * R_aft.transpose();
  }
  tran_diff = p_bef - rot_diff * p_aft;

  LOG(INFO) << "[4DOF] " << (apply ? "apply " : "not apply") << "; pos_diff  ["
            << VectorToString(tran_diff) << "]; y_diff: " << y_diff;

  if (!apply) {
    return;
  }

  SE3d SE3_Rt(rot_diff, tran_diff);
  int start_idx = trajectory_->GetCtrlIndex(t_min * S_TO_NS);
  int end_idx = trajectory_->GetCtrlIndex(t_max * S_TO_NS) + 3;
  for (int i = start_idx; i <= end_idx; i++) {
    trajectory_->setKnot(SE3_Rt * trajectory_->getKnot(i), i);
  }

  LOG(INFO) << "[4DOF] apply to  [" << start_idx << ", " << end_idx << "]";

  // Debug
  SE3d pose_aft = trajectory_->GetIMUPose(t_min);
  Eigen::Vector3d pos_aft = pose_aft.translation();
  euler_aft = R2ypr(pose_aft.so3().matrix());
  if ((p_bef - pos_aft).norm() > 1e-5) {
    LOG(INFO) << "[4DOF] desire time0 :" << t_min
              << "; p_bef: " << VectorToString(p_bef) << "; euler_bef "
              << VectorToString(euler_bef);
    LOG(INFO) << "[4DOF] aft time0    :" << t_min
              << "; p_aft: " << VectorToString(pos_aft) << "; euler_aft "
              << VectorToString(euler_aft);
  }
}

void TrajectoryManager::UpdateLiDARAttribute(double scan_time_min,
                                             double scan_time_max) {
  trajectory_->UpdateActiveTime(scan_time_max);
  // trajectory_->SetForcedFixedTime((scan_time_min + scan_time_max) / 2);
  trajectory_->SetForcedFixedTime(scan_time_min);

  if (trajectory_->maxTime(LiDARSensor) > 25) {
    double t = trajectory_->maxTime(LiDARSensor) - 25;
    RemoveIMUData(t);
  }
}

void TrajectoryManager::Log(std::string descri) const {
  IMUBias last_bias = all_imu_bias_.rbegin()->second;
  LOG(INFO) << descri
            << " Gyro Bias: " << VectorToString(last_bias.gyro_bias, 4)
            << "; Acce Bias: " << VectorToString(last_bias.accel_bias, 4);
  LOG(INFO) << "gravity: " << VectorToString(gravity_, 3);
  //  trajectory_->print_knots();
}

}  // namespace clic
