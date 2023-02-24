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

#include <Eigen/Geometry>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <eigen_conversions/eigen_msg.h>
#include <nav_msgs/Path.h>
#include <ros/ros.h>

#include <spline/trajectory.h>
#include <utils/parameter_struct.h>
#include <utils/eigen_utils.hpp>

namespace clic {

class GTLoader {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef std::shared_ptr<GTLoader> Ptr;

  GTLoader(std::string gt_file_in_tum = " ",
           bool loop_edge_init_from_gt = false)
      : has_data_(false),
        is_aligned_(false),
        loop_edge_init_from_gt_(loop_edge_init_from_gt) {
    pub_aligned_pose_ = nh_.advertise<nav_msgs::Path>("/gt_pose", 10);
    if (gt_file_in_tum != " ") {
      load_tum_data(gt_file_in_tum, pose_data_vec_);
      if (!pose_data_vec_.empty()) has_data_ = true;
    }
  }

  bool HasData() const { return has_data_; }

  bool IsAligned() const { return is_aligned_; }

  bool LoopEdgeInit() const { return loop_edge_init_from_gt_; }

  bool GetRelativeIMUPose(double t_target, double t_source, SE3d& T_target,
                          SE3d& T_source) const {
    if (!is_aligned_) return false;

    int idx[2] = {-1, -1};
    for (int i = 1; i < (int)pose_data_vec_aligned_.size(); ++i) {
      if (pose_data_vec_aligned_.at(i - 1).timestamp <= t_target &&
          t_target <= pose_data_vec_aligned_.at(i).timestamp) {
        idx[0] = i;
      }
      if (pose_data_vec_aligned_.at(i - 1).timestamp <= t_source &&
          t_source <= pose_data_vec_aligned_.at(i).timestamp) {
        idx[1] = i;
      }
      if (idx[0] > 0 && idx[1] > 0) break;
    }

    if (idx[0] < 0 || idx[1] < 0) return false;

    auto const& target_pose = pose_data_vec_aligned_.at(idx[0]);
    auto const& source_pose = pose_data_vec_aligned_.at(idx[1]);

    std::cout << "[GetRelativeIMUPose] qurey time: [" << t_target << ", "
              << t_source << "]\n";
    std::cout << "[GetRelativeIMUPose] get   time: [" << target_pose.timestamp
              << ", " << source_pose.timestamp << "]\n";

    T_target = SE3d(target_pose.orientation, target_pose.position);
    T_source = SE3d(source_pose.orientation, source_pose.position);
    return true;
  }

  void PublishAlignedPose(double traj_max_time = -1) {
    if (pub_aligned_pose_.getNumSubscribers() == 0) {
      is_aligned_ = false;
    }

    if (is_aligned_ && pub_aligned_pose_.getNumSubscribers() != 0) {
      ros::Time t_temp;
      std::vector<geometry_msgs::PoseStamped> poses_msg;
      double t_last = -1;
      for (auto const& v : pose_data_vec_aligned_) {
        if (v.timestamp - t_last > 0.1) {
          t_last = v.timestamp;
        } else {
          continue;
        }
        if (traj_max_time > 0 && v.timestamp > traj_max_time) break;

        geometry_msgs::PoseStamped geo_ctrl;
        geo_ctrl.header.stamp = t_temp.fromSec(v.timestamp);
        geo_ctrl.header.frame_id = "/map";
        tf::pointEigenToMsg(v.position, geo_ctrl.pose.position);
        tf::quaternionEigenToMsg(v.orientation.unit_quaternion(),
                                 geo_ctrl.pose.orientation);
        poses_msg.push_back(geo_ctrl);
      }

      nav_msgs::Path traj_ctrl;
      traj_ctrl.header.stamp = ros::Time::now();
      traj_ctrl.header.frame_id = "/map";
      traj_ctrl.poses = poses_msg;
      pub_aligned_pose_.publish(traj_ctrl);
    }
  }

  bool UmeyamaAlignment(const Trajectory::Ptr traj) {
    if (pub_aligned_pose_.getNumSubscribers() == 0) return false;

    double t_bag_start = traj->GetDataStartTime();
    double t_min = traj->minTime(IMUSensor) + t_bag_start;
    double t_max = traj->GetActiveTime() + t_bag_start;

    if ((t_max - t_bag_start) < 10) return false;

    for (auto iter = pose_data_vec_.begin(); iter != pose_data_vec_.end();) {
      if (iter->timestamp < t_bag_start - 0.1) {
        iter = pose_data_vec_.erase(iter);
      } else {
        break;
      }
    }

    if (pose_data_vec_.empty()) {
      has_data_ = false;
      return false;
    }

    std::vector<int> choosed_idx;
    double t_last = -1;
    for (int i = 0; i < (int)pose_data_vec_.size(); ++i) {
      double t_gt = pose_data_vec_.at(i).timestamp;
      if (t_gt < t_min) continue;
      if (t_gt >= t_max) break;
      if (t_gt - t_last > 0.1) {
        choosed_idx.push_back(i);
        t_last = t_gt;
      }
    }

    if (choosed_idx.size() < 100) return false;

    // can also check ATE with this alignment:
    Eigen::Matrix3Xd points_gt(3, choosed_idx.size());
    Eigen::Matrix3Xd points_traj(3, choosed_idx.size());
    for (int i = 0; i < (int)choosed_idx.size(); ++i) {
      auto const& pose_gt = pose_data_vec_.at(choosed_idx[i]);
      points_gt.col(i) = pose_gt.position;
      points_traj.col(i) =
          traj->GetPositionWorld(pose_gt.timestamp - t_bag_start);
    }

    // align trajectories using umeyama + fit found (cR+t)-transformation to R+t
    Eigen::Matrix4d T_result;
    bool with_scaling = false;
    T_result = Eigen::umeyama(points_gt, points_traj, with_scaling);
    SO3d Rot = SO3d(T_result.block<3, 3>(0, 0));
    Eigen::Vector3d pos = T_result.block<3, 1>(0, 3);

    pose_data_vec_aligned_.clear();
    pose_data_vec_aligned_.reserve(pose_data_vec_.size());
    for (auto const& v : pose_data_vec_) {
      PoseData pose_aligned;
      pose_aligned.timestamp = v.timestamp;
      pose_aligned.position = Rot * v.position + pos;
      pose_aligned.orientation = Rot * v.orientation;
      pose_data_vec_aligned_.push_back(pose_aligned);
    }

    std::cout << "UmeyamaAlignment success at " << traj->GetActiveTime()
              << "\n";
    is_aligned_ = true;
    return true;
  }

 public:
  static void load_tum_data(const std::string& gt_file_in_tum,
                            Eigen::aligned_vector<PoseData>& pose_data_vec) {
    Eigen::aligned_vector<Eigen::VectorXd> values;
    load_value_from_csv(gt_file_in_tum, values);

    assert(!values.empty() && values.front().size() == 8 &&
           "load_EuRoc_vicon_data failed.");

    for (auto const& v : values) {
      pose_data_vec.emplace_back();
      pose_data_vec.back().timestamp = v[0];  // in second
      pose_data_vec.back().position = v.segment<3>(1);

      Eigen::Quaterniond q(v[7], v[4], v[5], v[6]);  // wxyz
      pose_data_vec.back().orientation = SO3d(q);
    }
    std::cout << "Load " << pose_data_vec.size() << " pose data from "
              << gt_file_in_tum << std::endl;
  }

  static void load_value_from_csv(
      const std::string& path, Eigen::aligned_vector<Eigen::VectorXd>& values,
      char delim = ' ') {
    values.reserve(10000);
    // Try to open our trajectory file
    std::ifstream file(path);
    if (!file.is_open()) {
      std::cerr << "Unable to open file...  " << path << std::endl;
      std::exit(EXIT_FAILURE);
    }

    // Loop through each line of this file
    std::string current_line;
    while (std::getline(file, current_line)) {
      // Skip if we start with a comment
      if (!current_line.find("#")) continue;

      // Loop variables
      std::istringstream s(current_line);
      std::string field;
      std::vector<double> vec;

      // Loop through this line (timestamp(s) values....)
      while (std::getline(s, field, delim)) {
        // Skip if empty
        if (field.empty()) continue;
        // save the data to our vector
        vec.push_back(std::atof(field.c_str()));
      }

      // Create eigen vector
      Eigen::VectorXd temp(vec.size());
      for (size_t i = 0; i < vec.size(); i++) {
        temp(i) = vec.at(i);
      }
      values.push_back(temp);
    }

    // Finally close the file
    file.close();

    // Error if we don't have any data
    if (values.empty()) {
      std::cerr << "[LOAD]: Could not parse any data from the file " << path
                << "\n";
      std::exit(EXIT_FAILURE);
    }

    // Assert that all rows in this file are of the same length
    int rowsize = values.at(0).rows();
    for (size_t i = 0; i < values.size(); i++) {
      if (values.at(i).rows() != rowsize) {
        std::cerr << "[LOAD]: Invalid row size on line" << i << ".\n";
        std::exit(EXIT_FAILURE);
      }
    }
  }

 private:
  bool has_data_;
  bool is_aligned_;
  bool loop_edge_init_from_gt_;

  Eigen::aligned_vector<PoseData> pose_data_vec_;
  Eigen::aligned_vector<PoseData> pose_data_vec_aligned_;

  ros::NodeHandle nh_;
  ros::Publisher pub_aligned_pose_;
};

}  // namespace clic
