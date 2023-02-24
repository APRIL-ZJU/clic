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

#include <utils/parameter_struct.h>
#include <Eigen/Eigen>
#include <sophus_lib/so3.hpp>

namespace clic {

struct RelativePoseData {
  RelativePoseData()
      : target_timestamp(0),
        target_kf_index(0),
        source_timestamp(0),
        source_kf_index(0),
        position(Eigen::Vector3d(0, 0, 0)),
        orientation(SO3d(Eigen::Quaterniond::Identity())) {}
  double target_timestamp;
  int target_kf_index;
  double source_timestamp;
  int source_kf_index;
  Eigen::Vector3d position;
  SO3d orientation;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct VelocityData {
  VelocityData()
      : timestamp(0),
        velocity(Eigen::Vector3d(0, 0, 0)),
        gyro(Eigen::Vector3d(0, 0, 0)) {}
  double timestamp;
  Eigen::Vector3d velocity;
  Eigen::Vector3d gyro;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct LoopClosureOptimizationParam {
  double cur_search_start_timestamp;
  int cur_seach_start_index;

  double cur_timestamp;
  int cur_index;

  double history_timestamp;
  int history_index;

  double history_fix_time;
  int history_fix_index;

  double pose_graph_start_time;  // historical closure
  int pose_graph_start_index;  //  historical closure

  /// pose of closure
  RelativePoseData loop_closure_edge;

  std::vector<RelativePoseData> history_loop_closure_edges;

  std::vector<VelocityData> velocity_constraint;

  /// pose graph的pose
  std::vector<PoseData> pose_graph_key_pose;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct LoopClosureWeights {
  double pose_graph_edge_pos_weight;

  double pose_graph_edge_rot_weight;

  double velocity_weight;

  double gyro_weight;
};

}  // namespace clic
