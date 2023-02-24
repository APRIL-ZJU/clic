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

#include <eigen_conversions/eigen_msg.h>
#include <estimator/odometry_manager.h>

namespace clic {

OdometryManager::OdometryManager(const YAML::Node& node, ros::NodeHandle& nh)
    : odometry_mode_(LIO), is_initialized_(false) {
  std::string config_path = node["config_path"].as<std::string>();

  std::string lidar_yaml = node["lidar_odom_yaml"].as<std::string>();
  YAML::Node lidar_node = YAML::LoadFile(config_path + lidar_yaml);

  std::string imu_yaml = node["imu_yaml"].as<std::string>();
  YAML::Node imu_node = YAML::LoadFile(config_path + imu_yaml);

  odometry_mode_ = OdometryMode(node["odometry_mode"].as<int>());
  std::cout << YELLOW << "\t -odometry mode: ";
  if (odometry_mode_ == LICO) {
    std::cout << "LICO" << RESET << std::endl;
  } else if (odometry_mode_ == LIO) {
    std::cout << "LIO" << RESET << std::endl;
  }

  t_update_map = 0;
  t_update_traj = 0;
  t_update_prior = 0;

  ExtrinsicParam EP_LtoI, EP_CtoI, EP_ItoI;
  EP_LtoI.Init(lidar_node["lidar0"]["Extrinsics"]);
  // note: we consider lidar as the time baseline
  EP_LtoI.t_offset_ns = 0;
  
  if (odometry_mode_ == LICO) EP_CtoI.Init(node["CameraExtrinsics"]);
  // set time offset
  if (node["IMUExtrinsics"]) EP_ItoI.Init(node["IMUExtrinsics"]);

  double knot_distance = node["knot_distance"].as<double>();
  trajectory_ = std::make_shared<Trajectory>(knot_distance, 0);
  trajectory_->SetSensorExtrinsics(SensorType::LiDARSensor, EP_LtoI);
  trajectory_->SetSensorExtrinsics(SensorType::CameraSensor, EP_CtoI);
  trajectory_->SetSensorExtrinsics(SensorType::IMUSensor, EP_ItoI);

  update_every_k_knot_ = node["update_every_k_knot"].as<int>();

  // LiDAR Odometry
  lidar_odom_ = std::make_shared<LidarOdometry>(lidar_node, trajectory_);

  // Visual Odometry
  if (odometry_mode_ == LICO)
    visual_odom_ = std::make_shared<VisualOdometry>(node, trajectory_);
  else
    visual_odom_ == nullptr;

  loop_closure_ = std::make_shared<LoopClosure>(lidar_node, trajectory_);

  std::string gt_file = yaml::GetString(node, "tum_ground_truth_path", " ");
  bool loop_edge_init_from_gt = yaml::GetBool(node, "loop_edge_init_from_gt");
  gt_loader_ = std::make_shared<GTLoader>(gt_file, loop_edge_init_from_gt);

  // Trajectory And Others
  trajectory_manager_ = std::make_shared<TrajectoryManager>(node, trajectory_);

  imu_initializer_ = std::make_shared<IMUInitializer>(imu_node);

  odom_viewer_.SetPublisher(nh);

  msg_manager_ = std::make_shared<MsgManager>(node, nh);

  bool verbose;
  nh.param<double>("pasue_time", pasue_time_, -1);
  nh.param<bool>("verbose", verbose, false);
  trajectory_manager_->verbose = verbose;

  CreateCacheFolder(config_path, msg_manager_->bag_path_);

  sys_t_offset_vec_.reserve(10000);

  std::cout << std::fixed << std::setprecision(4);
  LOG(INFO) << std::fixed << std::setprecision(4);

  std::cout << YELLOW << "Time Offset init: \n" << RESET;
  std::cout << "\t IMUSensor time offset init: "
            << trajectory_->GetSensorEP(IMUSensor).t_offset_ns << " [ns]\n";
  std::cout << "\t CameraSensor time offset init: "
            << trajectory_->GetSensorEP(CameraSensor).t_offset_ns << " [ns]\n";
  std::cout << "\t LiDARSensor time offset init: "
            << trajectory_->GetSensorEP(LiDARSensor).t_offset_ns << " [ns]\n";

  LOG(INFO) << "\t IMUSensor time offset init: "
            << trajectory_->GetSensorEP(IMUSensor).t_offset_ns << " [ns]";
  LOG(INFO) << "\t CameraSensor time offset init: "
            << trajectory_->GetSensorEP(CameraSensor).t_offset_ns << " [ns]";
  LOG(INFO) << "\t LiDARSensor time offset init: "
            << trajectory_->GetSensorEP(LiDARSensor).t_offset_ns << " [ns]";
}

bool OdometryManager::CreateCacheFolder(const std::string& config_path,
                                        const std::string& bag_path) {
  boost::filesystem::path path_cfg(config_path);
  boost::filesystem::path path_bag(bag_path);
  if (path_bag.extension() != ".bag") {
    return false;
  }
  std::string bag_name_ = path_bag.stem().string();

  std::string cache_path_parent_ = path_cfg.parent_path().string();
  cache_path_ = cache_path_parent_ + "/data/" + bag_name_;
  boost::filesystem::create_directory(cache_path_parent_ + "/data/");
  return true;
}

void OdometryManager::RunInSubscribeMode() {
  ros::Rate rate(400);
  while (ros::ok()) {
    if (!is_initialized_) {
      while (!msg_manager_->imu_buf_.empty()) {
        imu_initializer_->FeedIMUData(msg_manager_->imu_buf_.front());
        msg_manager_->imu_buf_.pop_front();
      }

      if (imu_initializer_->StaticInitialIMUState()) {
        SetInitialState();
      }
    }

    // after initialization
    if (is_initialized_) {
      if (GetMsgsForProcessing()) {
        ProcessLICData();
      }
    }

    ros::spinOnce();
  }
}

void OdometryManager::RunBag() {
  LOG(INFO) << "RunBag ....";

  sys_t_offset_vec_.emplace_back(
      SysTimeOffset(0, trajectory_->GetSensorEP(LiDARSensor).t_offset_ns,
                    trajectory_->GetSensorEP(CameraSensor).t_offset_ns,
                    trajectory_->GetSensorEP(IMUSensor).t_offset_ns));
  while (ros::ok()) {
    msg_manager_->SpinBagOnce();
    if (!msg_manager_->has_valid_msg_) {
      break;
    }

    if (msg_manager_->t_image_ms_ > 0 || msg_manager_->t_lidar_ms_ > 0) {
      LOG(INFO) << "MsgHandler:" << trajectory_->maxTime(LiDARSensor) << ";"
                << msg_manager_->t_image_ms_ << ";"
                << msg_manager_->t_lidar_ms_;
    }

    if (!is_initialized_) {
      while (!msg_manager_->imu_buf_.empty()) {
        imu_initializer_->FeedIMUData(msg_manager_->imu_buf_.front());
        msg_manager_->imu_buf_.pop_front();
      }

      if (imu_initializer_->StaticInitialIMUState()) {
        SetInitialState();
      } else {
        continue;
      }
    }

    // after initialization
    if (GetMsgsForProcessing()) {
      ProcessLICData();

      sys_t_offset_vec_.emplace_back(
          SysTimeOffset(trajectory_->maxTime(LiDARSensor),
                        trajectory_->GetSensorEP(LiDARSensor).t_offset_ns,
                        trajectory_->GetSensorEP(CameraSensor).t_offset_ns,
                        trajectory_->GetSensorEP(IMUSensor).t_offset_ns));
    }
  }
}

void OdometryManager::ProcessLICData() {
  msg_manager_->LogInfo();

  if (is_initialized_) {
    time_summary_.ReStart();

    if (visual_odom_ && visual_odom_->IsWindowOk()) {
      visual_odom_->TriangulateLandmarks(false);
      time_summary_.TocAndTic("TriangulateLandmarks1");
    }

    if (!msg_manager_->next_msgs.image_feature_msgs.empty()) {
      LOG(INFO) << " ========== Update image ==========";
      ProcessImageData();

      time_summary_.TocAndTic("ProcessImageData");
    }

    bool proces_lidar = false;
    if (msg_manager_->next_msgs.lidar_timestamp > 0) {
      LOG(INFO) << " ========== Update lidar ========== ";
      ProcessLiDARData();
      proces_lidar = true;
    }

    if (loop_closure_->IsLoopClosureEnable(
            msg_manager_->next_msgs.lidar_timestamp)) {
      time_summary_.Tic();
      loop_closure_->LoopClosureHandler(lidar_odom_, gt_loader_);
      time_summary_.TocAndTic("LoopClosureHandler");

      if (loop_closure_->HasLoop()) {
        // clear prior
        trajectory_manager_->ClearLIOPrior();

        lidar_odom_->SetUpdateMap(true);

        odom_viewer_.PublishSplineType(trajectory_, SplineViewerType::Loop);
        // std::cout << "[loop closure] finish. Pause " << std::endl;
        // std::getchar();
      }
    }

    if (proces_lidar) {
      auto& msg = msg_manager_->next_msgs;

      TicToc timer;
      // update prior
      time_summary_.Tic();
      trajectory_manager_->UpdateLIOPrior(
          lidar_odom_->GetPointCorrespondence());
      t_update_prior += timer.toc() * 1e-3;
      time_summary_.TocAndTic("UpdateLIOPrior");

      trajectory_manager_->UpdateLiDARAttribute(msg.lidar_timestamp,
                                                msg.lidar_max_timestamp);

      if (visual_odom_ && visual_odom_->IsWindowOk()) {
        visual_odom_->SetDepth(trajectory_manager_->GetFeatureInvDepths());
        time_summary_.TocAndTic("SetDepth");

        if (MARGIN_OLD == visual_odom_->GetMarginalizationFlag()) {
          TicToc timer;
          trajectory_manager_->UpdateVisualOffsetPrior(
              visual_odom_->GetFeatures(), visual_odom_->GetGoodFeatureIDs(),
              visual_odom_->GetTimestamps());
          t_update_prior += timer.toc() * 1e-3;
          time_summary_.TocAndTic("UpdateVisualOffsetPrior");
        }
      }
    }
  }
  time_summary_.LogTimeStatistics(trajectory_->maxTime(LiDARSensor));

  /// visualization
  PublishCloudAndTrajectory();

  if (gt_loader_->HasData()) {
    if (!gt_loader_->IsAligned()) {
      gt_loader_->UmeyamaAlignment(trajectory_);
    }
    if (gt_loader_->IsAligned()) {
      gt_loader_->PublishAlignedPose(trajectory_->maxTime(LiDARSensor) +
                                     trajectory_->GetDataStartTime());
    }
  }

  // for debug
  if (pasue_time_ > 0 && trajectory_->maxTime(LiDARSensor) > pasue_time_) {
    // trajectory_->print_knots();
    std::cout << "Pause press any key to continue" << std::endl;
    std::getchar();
  }
}

void OdometryManager::ProcessLiDARData() {
  auto& msg = msg_manager_->next_msgs;
  LOG(INFO) << "Process " << msg.scan_num << " scans in ["
            << msg.lidar_timestamp << ", " << msg.lidar_max_timestamp << "]";

  msg.CheckData();

  time_summary_.Tic();
  lidar_odom_->FeatureCloudHandler(msg.lidar_timestamp, msg.lidar_max_timestamp,
                                   msg.lidar_corner_cloud, msg.lidar_surf_cloud,
                                   msg.lidar_raw_cloud);
  time_summary_.TocAndTic("FeatureHandler");

  /// [step2] : initialize traj
  trajectory_manager_->PropagateTrajectory(msg.lidar_timestamp,
                                           msg.lidar_max_timestamp);
  odom_viewer_.PublishSplineType(trajectory_, SplineViewerType::Init);
  time_summary_.TocAndTic("PropagateTraj");

  /// [step3] : update local map
  TicToc timer;
  lidar_odom_->UpdateMapData();
  t_update_map += timer.toc() * 1e-3;
  time_summary_.TocAndTic("UpdateMapData");

  for (int iter = 0; iter < 2; ++iter) {
    /// [step4] : Association
    lidar_odom_->GetLoamFeatureAssociation();
    time_summary_.TocAndTic("GetLoamFeatureAssociation" + std::to_string(iter));

    timer.tic();
    /// [step5] : update trajectory
    if (visual_odom_ && visual_odom_->IsWindowOk()) {
      trajectory_manager_->UpdateTrajectoryWithLoamFeature(
          lidar_odom_->GetPointCorrespondence(), visual_odom_->GetFeatures(),
          visual_odom_->GetGoodFeatureIDs(), visual_odom_->GetTimestamps(), 8);

      time_summary_.TocAndTic("UpdateTrajectoryLIC" + std::to_string(iter));
    } else {
      trajectory_manager_->UpdateTrajectoryWithLoamFeature(
          lidar_odom_->GetPointCorrespondence(), {}, {}, nullptr, 8);
      time_summary_.TocAndTic("UpdateTrajectoryLIO" + std::to_string(iter));
    }
  }
  t_update_traj += timer.toc() * 1e-3;

  auto bias = trajectory_manager_->GetLatestBias();
  odom_viewer_.PublishLatestBias(bias.gyro_bias, bias.accel_bias);
  trajectory_manager_->Log("[LIO]");

  auto pose = trajectory_->GetLidarPose(msg.lidar_timestamp);
  odom_viewer_.PublishTF(pose.unit_quaternion(), pose.translation(), "lidar",
                         "map");

  odom_viewer_.PublishTF(trajectory_manager_->GetGlobalFrame(),
                         Eigen::Vector3d::Zero(), "map", "global");
}

void OdometryManager::ProcessImageData() {
  double data_start_time = trajectory_->GetDataStartTime();
  if (data_start_time < 0) return;

  auto& msg = msg_manager_->next_msgs;
  if (msg.image_feature_msgs.empty()) return;

  for (const auto& img : msg.image_feature_msgs) {
    // leave a position for current frame
    visual_odom_->SlideWindow();

    visual_odom_->AddImageToWindow(img);
  }
}

bool OdometryManager::GetMsgsForProcessing() {
  if (!is_initialized_) return false;
  static double t_add = -1;

  double dt = trajectory_->getDt();
  double data_start_time = trajectory_->GetDataStartTime();

  bool in_scan_unit = false;

  if (t_add < 0) {
    t_add = update_every_k_knot_ * dt;
    if (in_scan_unit) {
      std::cout << YELLOW << "\n\t- Update traj at scan frequency\n" << RESET;
    } else {
      std::cout << YELLOW << "\n\t- Update traj every " << t_add << " second\n"
                << RESET;
    }
  }

  double traj_max_time;
  if (in_scan_unit) {
    traj_max_time = trajectory_->maxTime(LiDARSensor) + 0.21 + dt;
  } else {
    traj_max_time = trajectory_->maxTime(LiDARSensor) + t_add;
  }

  msg_manager_->t_offset_imu_ =
      trajectory_->GetSensorEP(IMUSensor).t_offset_ns * NS_TO_S;
  msg_manager_->t_offset_camera_ =
      trajectory_->GetSensorEP(CameraSensor).t_offset_ns * NS_TO_S;

  bool have_msg = msg_manager_->GetNextMsgs(traj_max_time, data_start_time, dt,
                                            in_scan_unit);

  if (have_msg) {
    while (!msg_manager_->imu_buf_.empty()) {
      trajectory_manager_->AddIMUData(msg_manager_->imu_buf_.front());
      msg_manager_->imu_buf_.pop_front();
    }
    return true;
  } else {
    return false;
  }
}

void OdometryManager::SetInitialState() {
  if (is_initialized_) {
    assert(trajectory_->GetDataStartTime() > 0 && "data start time < 0");
    std::cout << "[SetInitialState] system state has been initialized\n";
    return;
  }
  is_initialized_ = true;
  if (imu_initializer_->InitialDone()) {
    SystemState sys_state = imu_initializer_->GetIMUState();
    trajectory_manager_->SetSystemState(sys_state);

    /// set the start time of the trajectory
    trajectory_manager_->AddIMUData(imu_initializer_->GetIMUData().back());
    msg_manager_->imu_buf_.clear();

    std::cout << YELLOW
              << "use [imu_initializer] set trajectory start time as: "
              << trajectory_->GetDataStartTime() << RESET << std::endl;
  }
  assert(trajectory_->GetDataStartTime() > 0 && "data start time < 0");
}

void OdometryManager::PublishCloudAndTrajectory() {
  // visualize the inertial integraton and the initialized traj
  odom_viewer_.PublishIMUEstimator(trajectory_manager_->GetIMUStateEstimator());

  odom_viewer_.PublishDenseCloud(trajectory_, lidar_odom_->GetFeatureMapDs(),
                                 lidar_odom_->GetFeatureCurrentDs());
  odom_viewer_.PublishLoamMapCorrTemp(trajectory_,
                                      lidar_odom_->map_corrs_viewer);

  odom_viewer_.PublishSplineTrajectory(
      trajectory_, trajectory_->minTime(IMUSensor),
      trajectory_->maxTime(IMUSensor), trajectory_->getDt() * 2);

  odom_viewer_.PublishMargCtrlCloud(trajectory_manager_->GetMargCtrlPoint());

  odom_viewer_.PublishLoamCorrespondence(trajectory_,
                                         lidar_odom_->GetPointCorrespondence());

  if (visual_odom_ && visual_odom_->IsWindowOk()) {
    odom_viewer_.PublishTrackImage(visual_odom_->GetFeatures(),
                                   visual_odom_->GetGoodFeatureIDs(),
                                   msg_manager_->GetLatestImage());

    odom_viewer_.PublishVioKeyFrame(trajectory_, visual_odom_->GetTimestamps());

    VPointCloud landmarks_init, landmarks_opt;
    bool ret = visual_odom_->GetCurOptLandmarks(landmarks_init, landmarks_opt);
    if (ret) {
      odom_viewer_.PublishImageLandmarks(landmarks_init, landmarks_opt);
    }
  }
}

double OdometryManager::SaveOdometry() {
  std::string descri;
  if (odometry_mode_ == LICO)
    descri = "LIC";
  else if (odometry_mode_ == LIO)
    descri = "LIO";

  if (msg_manager_->NumLiDAR() > 1) descri = descri + "2";

  int knot_dist_ms = trajectory_->getDtNs() / 1e6;
  descri = descri + "_" + std::to_string(knot_dist_ms) + "_" +
           std::to_string(update_every_k_knot_);

  if (std::fabs(trajectory_->GetSensorEP(IMUSensor).t_offset_ns) > 1) {
    descri = descri + "_calib_" +
             std::to_string(int(msg_manager_->add_extra_timeoffset_s_ * 1000)) +
             "ms";
  } else if (std::fabs(msg_manager_->add_extra_timeoffset_s_) > 1e-9) {
    descri = descri + "_add_" +
             std::to_string(int(msg_manager_->add_extra_timeoffset_s_ * 1000)) +
             "ms";
  }

  ros::Time timer;
  std::string time_full_str = std::to_string(timer.now().toNSec());
  std::string t_str = "_" + time_full_str.substr(time_full_str.size() - 4);

  trajectory_->ToTUMTxt(cache_path_ + "_" + descri + "_odom" + t_str + ".txt",
                        0.01);

  // time offset
  if (std::fabs(trajectory_->GetSensorEP(IMUSensor).t_offset_ns) > 1) {
    std::ofstream outfile;
    outfile.open(cache_path_ + "_" + descri + "_offset" + t_str + ".txt");
    outfile.setf(std::ios::fixed);
    outfile << "t,t_lidar,t_cam,t_imu\n";
    for (auto const& data : sys_t_offset_vec_) {
      outfile.precision(9);
      outfile << data.timestamp << ",";
      outfile.precision(5);
      outfile << data.t_lidar << "," << data.t_cam << "," << data.t_imu << "\n";
    }
    outfile.close();
  }

  return trajectory_->maxTime(LiDARSensor);
}

}  // namespace clic
