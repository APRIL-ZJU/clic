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

#include <ros/package.h>
#include <ros/ros.h>

#include <estimator/odometry_manager.h>

using namespace clic;

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);

  ros::init(argc, argv, "clic");
  ros::NodeHandle nh("~");

  std::string config_path;
  nh.param<std::string>("config_path", config_path, "ct_odometry.yaml");
  ROS_INFO("Odometry load %s.", config_path.c_str());

  YAML::Node config_node = YAML::LoadFile(config_path);

  FLAGS_log_dir = "./log";
  FLAGS_colorlogtostderr = true;
  LOG(INFO) << "Start LiDAR Odometry";

  OdometryManager odom_manager(config_node, nh);

  MODE mode = MODE(config_node["mode"].as<int>());
  TicToc timer;
  if (MODE::Odometry_Offline == mode) {
    odom_manager.RunBag();
  } else if (MODE::Odometry_Online == mode) {
    LOG(INFO) << "RunInSubscribeMode ....";

    odom_manager.RunInSubscribeMode();
  }

  double t_traj_max = odom_manager.SaveOdometry();

  std::cout << "Total costs: " << timer.toc() * 1e-3
            << " [s]; traj max time:" << t_traj_max << " [s]\n";
  LOG(INFO) << "Total costs: " << timer.toc() * 1e-3
            << " [s]; traj max time:" << t_traj_max << " [s]";

  std::cout << "t_update_map: " << odom_manager.t_update_map
            << " [s]; t_update_traj:" << odom_manager.t_update_traj
            << " [s]; t_update_prior:" << odom_manager.t_update_prior
            << " [s]\n";

  return 0;
}
