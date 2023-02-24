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

#include <estimator/trajectory_estimator.h>
#include <loop_closure/loop_closure_data.h>
#include <loop_closure/pose_graph/pose_graph_3d_error_term.h>
#include <loop_closure/pose_graph/types.h>
#include <utils/parameter_struct.h>
#include <utils/eigen_utils.hpp>

#include <loop_closure/pose_graph.h>

namespace clic {

void PoseGraph::DiscretedPoseGraph(const PosCloud::Ptr& cloud_key_pos_xy,
                                   const Trajectory::Ptr& trajectory,
                                   LoopClosureOptimizationParam& param) {
  using namespace ceres::examples;
  MapOfPoses graph_poses;
  VectorOfConstraints graph_constraints;

  pose_vector_bef.clear();
  pose_data_after_opt.clear();

  std::vector<int> fix_index;
  int index = param.pose_graph_start_index;
  int cnt = 0;
  for (; index <= param.cur_index; index++) {
    double kf_time = cloud_key_pos_xy->points.at(index).timestamp;
    SE3d lidar_pose = trajectory->GetLidarPose(kf_time);
    pose_vector_bef.push_back(lidar_pose);
    Pose3d pose_temp;
    pose_temp.p = lidar_pose.translation();
    pose_temp.q = lidar_pose.unit_quaternion();
    graph_poses.insert(std::make_pair(cnt, pose_temp));
    if (index >= param.history_index && index <= param.history_fix_index) {
      fix_index.push_back(cnt);
    }
    cnt++;
  }

  // TODO the select of the latest pose
  // double tra_max_time = feature_cur_.time_max;
  double tra_max_time =
      trajectory->maxTime(LiDARSensor) - trajectory->getDt() / 5;
  SE3d last_lidar_pose = trajectory->GetLidarPose(tra_max_time);
  pose_vector_bef.push_back(last_lidar_pose);
  Pose3d last_pose;
  last_pose.p = last_lidar_pose.translation();
  last_pose.q = last_lidar_pose.unit_quaternion();
  graph_poses.insert(std::make_pair(cnt, last_pose));

  // edge: scan(i+1) to (i)
  for (size_t i = 0; i < pose_vector_bef.size() - 1; i++) {
    Constraint3d temp;
    temp.id_begin = i;
    temp.id_end = i + 1;
    temp.information << 10000, 0, 0, 0, 0, 0,  //
        0, 10000, 0, 0, 0, 0,                  //
        0, 0, 10000, 0, 0, 0,                  //
        0, 0, 0, 20000, 0, 0,                  //
        0, 0, 0, 0, 20000, 0,                  //
        0, 0, 0, 0, 0, 20000;                  //
    temp.information *= 3;
    SE3d delta_pose = pose_vector_bef[i].inverse() * pose_vector_bef[i + 1];
    temp.t_be.p = delta_pose.translation();
    temp.t_be.q = delta_pose.unit_quaternion();
    graph_constraints.push_back(temp);
  }

  // edge: current scan to history scan
  Constraint3d temp;
  temp.id_begin = param.history_index - param.pose_graph_start_index;
  temp.id_end = param.cur_index - param.pose_graph_start_index;
  temp.information << 10000, 0, 0, 0, 0, 0,  //
      0, 10000, 0, 0, 0, 0,                  //
      0, 0, 10000, 0, 0, 0,                  //
      0, 0, 0, 20000, 0, 0,                  //
      0, 0, 0, 0, 20000, 0,                  //
      0, 0, 0, 0, 0, 20000;                  //
  temp.t_be.p = param.loop_closure_edge.position;
  temp.t_be.q = param.loop_closure_edge.orientation.unit_quaternion();
  graph_constraints.push_back(temp);

  for (size_t i = 0; i < param.history_loop_closure_edges.size(); i++) {
    Constraint3d temp;
    temp.id_begin = param.history_loop_closure_edges[i].target_kf_index -
                    param.pose_graph_start_index;
    temp.id_end = param.history_loop_closure_edges[i].source_kf_index -
                  param.pose_graph_start_index;
    temp.information << 10000, 0, 0, 0, 0, 0,  //
        0, 10000, 0, 0, 0, 0,                  //
        0, 0, 10000, 0, 0, 0,                  //
        0, 0, 0, 10000, 0, 0,                  //
        0, 0, 0, 0, 10000, 0,                  //
        0, 0, 0, 0, 0, 10000;                  //
    temp.t_be.p = param.history_loop_closure_edges[i].position;
    temp.t_be.q =
        param.history_loop_closure_edges[i].orientation.unit_quaternion();
    graph_constraints.push_back(temp);
  }

  /// Optimization
  ceres::Problem problem;

  ceres::LossFunction* loss_function = NULL;
  ceres::LocalParameterization* quaternion_local_parameterization =
      new ceres::EigenQuaternionParameterization;

  for (VectorOfConstraints::const_iterator constraints_iter =
           graph_constraints.begin();
       constraints_iter != graph_constraints.end(); ++constraints_iter) {
    const Constraint3d& constraint = *constraints_iter;

    MapOfPoses::iterator pose_begin_iter =
        graph_poses.find(constraint.id_begin);
    MapOfPoses::iterator pose_end_iter = graph_poses.find(constraint.id_end);

    const Eigen::Matrix<double, 6, 6> sqrt_information =
        constraint.information.llt().matrixL();
    // Ceres will take ownership of the pointer.
    ceres::CostFunction* cost_function =
        PoseGraph3dErrorTerm::Create(constraint.t_be, sqrt_information);

    problem.AddResidualBlock(cost_function, loss_function,
                             pose_begin_iter->second.p.data(),
                             pose_begin_iter->second.q.coeffs().data(),
                             pose_end_iter->second.p.data(),
                             pose_end_iter->second.q.coeffs().data());

    problem.SetParameterization(pose_begin_iter->second.q.coeffs().data(),
                                quaternion_local_parameterization);
    problem.SetParameterization(pose_end_iter->second.q.coeffs().data(),
                                quaternion_local_parameterization);
  }

  for (size_t i = 0; i < fix_index.size(); i++) {
    int index = fix_index[i];
    MapOfPoses::iterator iter = graph_poses.find(index);
    if (iter != graph_poses.end()) {
      problem.SetParameterBlockConstant(iter->second.p.data());
      problem.SetParameterBlockConstant(iter->second.q.coeffs().data());
    }
  }

  ceres::Solver::Options options;
  options.max_num_iterations = 20;
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  options.minimizer_progress_to_stdout = false;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  std::cout << summary.BriefReport() << std::endl;

  //  PosCloud full_cloud;
  for (int id = param.pose_graph_start_index; id <= param.cur_index; id++) {
    int pose_index = id - param.pose_graph_start_index;
    auto pose_iter = graph_poses.find(pose_index);
    if (pose_iter == graph_poses.end()) continue;

    PoseData pd;
    pd.timestamp = cloud_key_pos_xy->points.at(id).timestamp;
    pd.position = pose_iter->second.p;
    pd.orientation = Sophus::SO3d(pose_iter->second.q);
    param.pose_graph_key_pose.push_back(pd);

    pose_data_after_opt.push_back(SE3d(pd.orientation, pd.position));
  }

  // add the latest pose
  auto pose_iter = graph_poses.find(cnt);
  PoseData pd;
  pd.timestamp = tra_max_time;
  pd.position = pose_iter->second.p;
  pd.orientation = Sophus::SO3d(pose_iter->second.q);
  param.pose_graph_key_pose.push_back(pd);
  pose_data_after_opt.push_back(SE3d(pd.orientation, pd.position));

  {
    int cur_id = param.cur_index - param.pose_graph_start_index;
    auto pose_iter = graph_poses.find(cur_id);
    cur_pose_aft_discrete_pg = SE3d(pose_iter->second.q, pose_iter->second.p);
  }
}

void PoseGraph::ContinuousPoseGraph(const LoopClosureOptimizationParam& param,
                                    const LoopClosureWeights& weights,
                                    Trajectory::Ptr trajectory) {
// #define Six_DoF_transform
#ifdef Six_DoF_transform
  double t_cur_keyscan = param.cur_timestamp;
  SE3d pose_bef = trajectory->GetIMUPose(t_cur_keyscan);
  int start_idx = trajectory->GetCtrlIndex(t_cur_keyscan * S_TO_NS);

  Eigen::aligned_map<int, SE3d> se3_knots;
  for (int i = start_idx; i < (int)trajectory->numKnots(); ++i) {
    se3_knots[i] = trajectory->getKnot(i);
  }
#endif

  TrajectoryEstimatorOptions option;
  option.lock_ab = true;
  option.lock_wb = true;
  option.lock_g = true;
  option.use_auto_diff = true;

  TrajectoryEstimator::Ptr estimator(
      new TrajectoryEstimator(trajectory, option));

  // fix the oldest ctrl points
  int fix_idx =
      trajectory->GetCtrlIndex(param.history_fix_time * S_TO_NS) + SplineOrder;
  estimator->SetFixedIndex(fix_idx);
  trajectory->opt_min_loop_time_tmp = param.pose_graph_start_time;
  trajectory->opt_loop_fixed_idx_tmp = fix_idx;
  std::cout << "[ContinuousPoseGraph] fixed [~, " << fix_idx << "]\n";

  for (const auto& v : param.velocity_constraint) {
    Eigen::Matrix<double, 6, 1> vel;
    vel.block<3, 1>(0, 0) = v.gyro;
    vel.block<3, 1>(3, 0) = v.velocity;
    estimator->Add6DofLocalVelocityAutoDiff(
        v.timestamp, vel, weights.gyro_weight, weights.velocity_weight);
  }

  // IMU pose
  auto EP_LtoI = trajectory->GetSensorEP(LiDARSensor);
  SO3d S_ItoL = EP_LtoI.so3.inverse();
  Eigen::Vector3d p_IinL = S_ItoL * (-EP_LtoI.p);
  for (const auto& pose_LtoG : param.pose_graph_key_pose) {
    PoseData pose_ItoG;
    // ItoG = LtoG * ItoL
    pose_ItoG.timestamp = pose_LtoG.timestamp;
    pose_ItoG.orientation = pose_LtoG.orientation * S_ItoL;
    pose_ItoG.position = pose_LtoG.orientation * p_IinL + pose_LtoG.position;

    estimator->AddPoseMeasurementAutoDiff(pose_ItoG,
                                          weights.pose_graph_edge_pos_weight,
                                          weights.pose_graph_edge_rot_weight);
  }

  ceres::Solver::Summary summary = estimator->Solve(50, false);
  std::cout << summary.BriefReport() << std::endl;

#ifdef Six_DoF_transform
  SE3d pose_aft = trajectory->GetIMUPose(t_cur_keyscan);  // A
  // A0_to_G * (B0_to_G)^T
  SE3d SE3_diff = pose_aft * pose_bef.inverse();
  // A0_to_G * (B0_to_G)^T * Bi_to_G
  for (auto const& v : se3_knots) {
    trajectory->setKnot(SE3_diff * v.second, v.first);
  }

  std::cout << "[Traj 6Dof transform] [" << start_idx << ", "
            << trajectory->numKnots() - 1 << "]\n";
#endif
  // auto const& knot = trajectory->getKnot(trajectory->numKnots() - 2);
  // trajectory->setKnot(knot, trajectory->numKnots() - 1);
}

}  // namespace clic
