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

#include <loop_closure/loop_closure_data.h>
#include <spline/trajectory.h>
#include <utils/mypcl_cloud_type.h>
#include <utils/parameter_struct.h>

namespace clic {

class PoseGraph {
 public:
  PoseGraph() {}

  std::vector<SE3d> pose_vector_bef;
  std::vector<SE3d> pose_data_after_opt;

  SE3d cur_pose_aft_discrete_pg;

  void DiscretedPoseGraph(const PosCloud::Ptr& cloud_key_pos_xy,
                          const Trajectory::Ptr& trajectory,
                          LoopClosureOptimizationParam& param);

  void ContinuousPoseGraph(const LoopClosureOptimizationParam& param,
                           const LoopClosureWeights& weights,
                           Trajectory::Ptr trajectory);
};

}  // namespace clic