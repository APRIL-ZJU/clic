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

#include <ros/ros.h>

#include <estimator/msg_manager.h>
#include <estimator/odometry_viewer.h>
#include <estimator/trajectory_manager.h>

#include <inertial/imu_state_estimator.h>
#include <inertial/inertial_initializer.h>
#include <lidar_odometry/lidar_odometry.h>
#include <loop_closure/gt_loader.h>
#include <loop_closure/loop_closure.h>
#include <visual_odometry/visual_odometry.h>

#include <condition_variable>
#include <mutex>
#include <thread>

namespace clic {

class OdometryManager {
 public:
  OdometryManager(const YAML::Node& node, ros::NodeHandle& nh);

  void RunBag();

  void RunInSubscribeMode();

  double SaveOdometry();

  double t_update_map, t_update_traj, t_update_prior;

 protected:
  bool CreateCacheFolder(const std::string& config_path,
                         const std::string& bag_path);

  void ProcessLICData();

  void ProcessLiDARData();

  void ProcessImageData();

  bool GetMsgsForProcessing();

  void SetInitialState();

  void PublishCloudAndTrajectory();

 protected:
  OdometryMode odometry_mode_;

  MsgManager::Ptr msg_manager_;

  bool is_initialized_;
  IMUInitializer::Ptr imu_initializer_;

  Trajectory::Ptr trajectory_;
  TrajectoryManager::Ptr trajectory_manager_;

  LidarOdometry::Ptr lidar_odom_;

  VisualOdometry::Ptr visual_odom_;

  LoopClosure::Ptr loop_closure_;

  GTLoader::Ptr gt_loader_;

  OdometryViewer odom_viewer_;

  int update_every_k_knot_;

  // save the results
  std::string cache_path_;

  // for debug
  double pasue_time_;

  TimeStatistics time_summary_;

  struct SysTimeOffset {
    SysTimeOffset(double t1, double t2, double t3, double t4)
        : timestamp(t1), t_lidar(t2), t_cam(t3), t_imu(t4) {}
    double timestamp = 0;
    double t_lidar = 0;
    double t_cam = 0;
    double t_imu = 0;
  };
  std::vector<SysTimeOffset> sys_t_offset_vec_;
};

}  // namespace clic
