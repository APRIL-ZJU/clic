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

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_broadcaster.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <pcl/kdtree/impl/kdtree_flann.hpp>  // pcl::KdTreeFLANN

#include <pcl/common/transforms.h>  //pcl::transformPointCloud
#include <pcl/registration/icp.h>
#include <pcl_conversions/pcl_conversions.h>  //pcl::fromROSMsg

#include <loop_closure/loop_closure.h>

namespace clic {

LoopClosure::LoopClosure(const YAML::Node& node, Trajectory::Ptr traj)
    : has_loop_(false),
      trajectory_(traj),
      t_last_successed_loop_(-1),
      kdtree_history_key_poses_(new pcl::KdTreeFLANN<PosPoint>()) {
  lc_param.Init(node);

  loop_closure_enable_flag_ = node["loop_closure_enable_flag"].as<bool>();
  loop_closure_frequency_ = node["loop_closure_frequency"].as<double>();

  const auto& w_node = node["loop_closure_weights"];

  lc_weights.velocity_weight = w_node["velocity_weight"].as<double>();
  lc_weights.gyro_weight = w_node["gyro_weight"].as<double>();
  lc_weights.pose_graph_edge_pos_weight =
      w_node["pose_graph_edge_pos_weight"].as<double>();
  lc_weights.pose_graph_edge_rot_weight =
      w_node["pose_graph_edge_rot_weight"].as<double>();

  printf("Loop Closure Weight:\n");
  printf("\t- velocity_weight: %.1f\n", lc_weights.velocity_weight);
  printf("\t- gyro_weight: %.1f\n", lc_weights.gyro_weight);
  printf("\t- pos_weight: %.1f\n", lc_weights.pose_graph_edge_pos_weight);
  printf("\t- rot_weight: %.1f\n", lc_weights.pose_graph_edge_rot_weight);

  const double& leaf_size = lc_param.loop_closure_downsample_leaf_size;
  loop_closure_match_voxel_filter_.setLeafSize(leaf_size, leaf_size, leaf_size);

  nh_ = ros::NodeHandle("~");

  /// publish icp cloud
  pub_icp_target_cloud_ =
      nh_.advertise<sensor_msgs::PointCloud2>("/loop/icp_target_cloud", 10);
  pub_icp_source_cloud_ =
      nh_.advertise<sensor_msgs::PointCloud2>("/loop/icp_source_cloud", 10);
  pub_icp_guess_source_cloud_ = nh_.advertise<sensor_msgs::PointCloud2>(
      "/loop/icp_guess_source_cloud", 10);
  pub_pose_graph_marker_ = nh_.advertise<visualization_msgs::MarkerArray>(
      "/loop/pose_graph_markers", 10);

  pub_loop_closure_marker_ = nh_.advertise<visualization_msgs::MarkerArray>(
      "/loop/clousre_markers", 10);
}

void LoopClosure::LoopClosureHandler(LidarOdometry::Ptr lidar_odom_,
                                     const GTLoader::Ptr gt_loader) {
  has_loop_ = false;

  cloud_key_pos_xy_ = lidar_odom_->GetCloudKeyPos2D();

  std::pair<int, int> cur_wrt_history;
  if (!DetectLoopClosure(cloud_key_pos_xy_, cur_wrt_history)) return;
  has_loop_ = true;

  double t_cur = cloud_key_pos_xy_->at(cur_wrt_history.first).timestamp;
  double t_history = cloud_key_pos_xy_->at(cur_wrt_history.second).timestamp;
  std::cout << "[Loop Closure] scan index: " << t_cur << " <---> " << t_history
            << " / time: " << t_cur << " <---> " << t_history
            << " ===> traj max " << trajectory_->maxTime(LiDARSensor)
            << std::endl;

  // loop edge from point registration
  Eigen::Matrix4f corrected_T_cur_to_history;
  {
    // select nearby scans
    // std::cout << "[Loop Closure]  Get local map...\n";
    VPointCloud::Ptr cur_vcloud(new VPointCloud);
    VPointCloud::Ptr history_vcloud(new VPointCloud);

    PosCloud::Ptr cur_cloud(new PosCloud);
    PosCloud::Ptr history_cloud(new PosCloud);
    lidar_odom_->FindNearbyKeyFrames(cur_wrt_history.first,
                                     lc_param.scan_search_num, cur_cloud);
    lidar_odom_->FindNearbyKeyFrames(cur_wrt_history.second,
                                     lc_param.scan_search_num, history_cloud);

    pcl::PosCloudToVPointCloud(*cur_cloud, *cur_vcloud);
    pcl::PosCloudToVPointCloud(*history_cloud, *history_vcloud);

    DownsampleLocalMap(cur_vcloud);
    DownsampleLocalMap(history_vcloud);

    Eigen::Matrix4d T_G_to_cur =
        trajectory_->GetLidarPose(t_cur).inverse().matrix();
    Eigen::Matrix4d T_G_to_his =
        trajectory_->GetLidarPose(t_history).inverse().matrix();

    VPointCloud::Ptr cur_vcloud_in_local(new VPointCloud);
    VPointCloud::Ptr history_vcloud_in_local(new VPointCloud);

    pcl::transformPointCloud(*cur_vcloud, *cur_vcloud_in_local,
                             T_G_to_cur.cast<float>());
    pcl::transformPointCloud(*history_vcloud, *history_vcloud_in_local,
                             T_G_to_his.cast<float>());

    Eigen::Matrix4f guess_T_cur_to_history;
    guess_T_cur_to_history = (T_G_to_his * T_G_to_cur.inverse()).cast<float>();
    if (gt_loader->IsAligned() && gt_loader->LoopEdgeInit()) {
      gt_loader->UmeyamaAlignment(trajectory_);
      SE3d gt_target;
      SE3d gt_source;
      double t_bag_start = trajectory_->GetDataStartTime();
      if (gt_loader->GetRelativeIMUPose(t_history + t_bag_start,
                                        t_cur + t_bag_start, gt_target,
                                        gt_source)) {
        std::cout << "traj pose\n";
        std::cout << "target_pose: "
                  << trajectory_->GetPositionWorld(t_history).transpose()
                  << "\nsource_pose: "
                  << trajectory_->GetPositionWorld(t_cur).transpose() << "\n";
        std::cout << "gt pose\n";
        std::cout << "target_pose: " << gt_target.translation().transpose()
                  << "\n";
        std::cout << "source_pose: " << gt_source.translation().transpose()
                  << "\n";

        SE3d T_lidar_cur_to_his;
        {
          auto pose_cur = trajectory_->GetIMUPose(t_cur);
          auto pose_his = trajectory_->GetIMUPose(t_history);
          pose_cur.translation() = gt_source.translation();
          pose_his.translation() = gt_target.translation();
          auto pose_imu_cur_to_his = pose_his.inverse() * pose_cur;
          auto const& EP_LtoI = trajectory_->GetSensorEP(LiDARSensor);
          T_lidar_cur_to_his =
              EP_LtoI.se3.inverse() * pose_imu_cur_to_his * EP_LtoI.se3;
        }

        std::cout << YELLOW << "guess_T_cur_to_history from gt_loader: \n"
                  << T_lidar_cur_to_his.matrix()
                  << "\nguess_T_cur_to_history from trajectory: \n"
                  << guess_T_cur_to_history << "\n"
                  << RESET;

        guess_T_cur_to_history = T_lidar_cur_to_his.matrix().cast<float>();
      }
    }

    // std::cout << "[Loop Closure]  Get loop pose...\n";
    if (!GetLoopPose(cur_wrt_history, cur_vcloud_in_local,
                     history_vcloud_in_local, guess_T_cur_to_history,
                     corrected_T_cur_to_history))
      return;
  }

  // record the time offset
  double t_offset_ns_imu = trajectory_->GetSensorEP(IMUSensor).t_offset_ns;
  trajectory_->GetSensorEPs().at(IMUSensor).t_offset_ns = 0;

  // std::cout << "[Loop Closure] PerformPoseGraph...\n";
  PerformPoseGraph(cur_wrt_history, corrected_T_cur_to_history);

  trajectory_->GetSensorEPs().at(IMUSensor).t_offset_ns = t_offset_ns_imu;

  // update key_pose
  lidar_odom_->UpdateCloudKeyPos(cur_wrt_history);

  // record the loop timestamp
  t_last_successed_loop_ = t_cur;

  PublishLoopClosureMarkers(lidar_odom_->GetCloudKeyPos3D());
}


bool LoopClosure::DetectLoopClosure(PosCloud::Ptr cloud_key_pos_xy_in,
                                    std::pair<int, int>& cur_wrt_history) {
  if (cloud_key_pos_xy_in->points.size() == 0) return false;
  cloud_key_pos_xy_ = cloud_key_pos_xy_in;

  int cur_idx = cloud_key_pos_xy_->size() - 1;
  int history_idx = -1;
  double cur_timestamp = cloud_key_pos_xy_->points.at(cur_idx).timestamp;

  if (history_loop_info_.find(cur_idx) != history_loop_info_.end()) {
    return false;
  }

  std::vector<int> loop_search_index;
  std::vector<float> loop_search_distance;
  kdtree_history_key_poses_->setInputCloud(cloud_key_pos_xy_);
  kdtree_history_key_poses_->radiusSearch(
      cloud_key_pos_xy_->back(), lc_param.scan_search_radius, loop_search_index,
      loop_search_distance);
  for (int i = 0; i < (int)loop_search_index.size(); i++) {
    int index = loop_search_index[i];
    double time = cloud_key_pos_xy_->points.at(index).timestamp;
    if (std::fabs(time - cur_timestamp) > lc_param.scan_time_diff &&
        (cur_idx - index) > lc_param.scan_index_diff) {
      history_idx = index;
      break;
    }
  }

  if (history_idx < 0 || cur_idx == history_idx) {
    return false;
  }

  cur_wrt_history.first = cur_idx;
  cur_wrt_history.second = history_idx;
  return true;
}

bool LoopClosure::PerformPoseGraph(
    std::pair<int, int>& cur_wrt_history,
    const Eigen::Matrix4f& corrected_T_cur_to_history) {
  double t_cur = cloud_key_pos_xy_->points[cur_wrt_history.first].timestamp;
  double t_history =
      cloud_key_pos_xy_->points[cur_wrt_history.second].timestamp;

  LoopClosureOptimizationParam lco_param;
  // current scan
  lco_param.cur_index = cur_wrt_history.first;
  lco_param.cur_timestamp = t_cur;
  // history scan
  lco_param.history_index = cur_wrt_history.second;
  lco_param.history_timestamp = t_history;

  // involved scan
  lco_param.cur_seach_start_index =
      lco_param.cur_index - lc_param.scan_search_num;
  lco_param.cur_search_start_timestamp =
      cloud_key_pos_xy_->points[lco_param.cur_seach_start_index].timestamp;

  lco_param.history_fix_index =
      lco_param.history_index + lc_param.scan_search_num;
  lco_param.history_fix_time =
      cloud_key_pos_xy_->points[lco_param.history_fix_index].timestamp;

  SE3d lidar_relative_pose = Matrix4fToSE3d(corrected_T_cur_to_history);

  auto& lc_edge = lco_param.loop_closure_edge;
  lc_edge.target_timestamp = lco_param.history_timestamp;
  lc_edge.target_kf_index = lco_param.history_index;
  lc_edge.source_timestamp = lco_param.cur_timestamp;
  lc_edge.source_kf_index = lco_param.cur_index;
  lc_edge.position = lidar_relative_pose.translation();
  lc_edge.orientation = lidar_relative_pose.so3();

  // pose bef loop closure
  SE3d T_cur_to_G = trajectory_->GetLidarPose(t_cur);
  SE3d T_his_to_G = trajectory_->GetLidarPose(t_history);

  ComputeLoopClosureParam(lco_param);

  // pose graph
  pose_graph_node.DiscretedPoseGraph(cloud_key_pos_xy_, trajectory_, lco_param);

  // visualize the pose graph
  PublishDiscretedPoseGraphMarker(pose_graph_node.pose_vector_bef,
                                  pose_graph_node.pose_data_after_opt);

  // continuous pose graph
  pose_graph_node.ContinuousPoseGraph(lco_param, lc_weights, trajectory_);

  history_loop_info_[cur_wrt_history.first] = cur_wrt_history.second;
  history_loop_edgs_[cur_wrt_history.first] = lco_param.loop_closure_edge;

  std::cout << GREEN << " ======== Loop Closure Result ======== \n";
  SE3d T_his_to_G_aft = trajectory_->GetLidarPose(lco_param.history_timestamp);
  SE3d T_cur_to_G_aft = trajectory_->GetLidarPose(lco_param.cur_timestamp);
  Eigen::Vector3d his_bef_p, his_bef_r, cur_bef_p, cur_bef_r;
  Eigen::Vector3d his_aft_p, his_aft_r, cur_aft_p, cur_aft_r;
  SE3dToPositionEuler(T_his_to_G, his_bef_p, his_bef_r);
  SE3dToPositionEuler(T_cur_to_G, cur_bef_p, cur_bef_r);
  SE3dToPositionEuler(T_his_to_G_aft, his_aft_p, his_aft_r);
  SE3dToPositionEuler(T_cur_to_G_aft, cur_aft_p, cur_aft_r);

  Eigen::Vector3d p_DLC, r_DLC;
  SE3dToPositionEuler(pose_graph_node.cur_pose_aft_discrete_pg, p_DLC, r_DLC);

  std::cout << lco_param.history_timestamp << " <-----> "
            << lco_param.cur_timestamp << std::endl;
  std::cout << "Before LC History pose: " << his_bef_p.transpose() << " --- "
            << his_bef_r.transpose() << std::endl;
  std::cout << "Before LC Current pose: " << cur_bef_p.transpose() << " --- "
            << cur_bef_r.transpose() << std::endl;
  std::cout << "After DLC Current pose: " << p_DLC.transpose() << " --- "
            << r_DLC.transpose() << std::endl;
  std::cout << "After CLC Current pose: " << cur_aft_p.transpose() << " --- "
            << cur_aft_r.transpose() << std::endl;
  std::cout << "-------------------------------------" << RESET << std::endl;

  return true;
}

// estimate the loop edge
bool LoopClosure::GetLoopPose(const std::pair<int, int>& cur_wrt_history,
                              VPointCloud::Ptr cur_cloud,
                              VPointCloud::Ptr history_cloud,
                              Eigen::Matrix4f& guess_T_cur_to_history,
                              Eigen::Matrix4f& corrected_T_cur_to_history) {
  double t_cur = cloud_key_pos_xy_->points[cur_wrt_history.first].timestamp;
  double t_history =
      cloud_key_pos_xy_->points[cur_wrt_history.second].timestamp;
  Eigen::Matrix4d T_cur_to_G = trajectory_->GetLidarPose(t_cur).matrix();
  Eigen::Matrix4d T_his_to_G = trajectory_->GetLidarPose(t_history).matrix();

  double delta_height = T_cur_to_G(2, 3) - T_his_to_G(2, 3);

  float fitness_score;
  while (true) {
    bool icp_success = PointToPointICP(
        history_cloud, cur_cloud, delta_height * 2, guess_T_cur_to_history,
        corrected_T_cur_to_history, fitness_score);

    PublishICPCloud(history_cloud, cur_cloud, T_his_to_G.cast<float>(),
                    corrected_T_cur_to_history, guess_T_cur_to_history);

    if (!lc_param.icp_manual_check) {
      break;

      if (icp_success) {
        break;
      } else {
        return false;
      }
    }

    if (lc_param.icp_manual_check) {
      //      if (icp_success) break;
      std::cout << RED << " Is ICP registration successful ? (Y or N)" << RESET
                << std::endl;
      char con;

      std::cin >> con;
      if (con == 'y' || con == 'Y') {
        std::cout << GREEN << " ICP score " << fitness_score << RESET
                  << std::endl;
        break;
      } else if (con == 'n' || con == 'N') {
        std::cout << "Please input (guess_cur_to_history) dx , dy , dz : "
                  << std::endl;
        float dx = 0, dy = 0, dz = 0;
        while (!(std::cin >> dx >> dy >> dz)) {
          std::cin.clear();
          std::cin.ignore();
          std::cin.sync();
          std::cout << "Input Error dxyz" << std::endl;
        }
        // guess_T_cur_to_history = corrected_T_cur_to_history;
        guess_T_cur_to_history(0, 3) += dx;
        guess_T_cur_to_history(1, 3) += dy;
        guess_T_cur_to_history(2, 3) += dz;

      } else if (con == 'r' || con == 'R') {
        return false;
      } else {
        std::cout << "Input Error ! Input y[yes] or n[no] or r[reject]\n";
      }
    }
  }
  return true;
}

// history_cloud, cur_cloud
bool LoopClosure::PointToPointICP(const VPointCloud::Ptr& target_cloud,
                                  const VPointCloud::Ptr& source_cloud,
                                  double correspondence_distance,
                                  Eigen::Matrix4f guess,
                                  Eigen::Matrix4f& transform,
                                  float& fitness_score) {
  if (correspondence_distance < 1) correspondence_distance = 1;
  pcl::IterativeClosestPoint<VPoint, VPoint> icp;
  icp.setMaxCorrespondenceDistance(correspondence_distance);
  icp.setMaximumIterations(100);
  icp.setTransformationEpsilon(1e-6);
  icp.setEuclideanFitnessEpsilon(1e-6);
  icp.setRANSACIterations(0);

  icp.setInputSource(source_cloud);
  icp.setInputTarget(target_cloud);

  VPointCloud result_cloud;
  icp.align(result_cloud, guess);

  fitness_score = icp.getFitnessScore();
  transform = icp.getFinalTransformation();

  std::cout << GREEN << "ICP score : " << fitness_score
            << " Coverged : " << icp.hasConverged() << RESET << std::endl;

  if (icp.hasConverged() == false ||
      fitness_score > lc_param.scan_fitness_score) {
    return false;
  }

  return true;
}

void LoopClosure::ComputeLoopClosureParam(LoopClosureOptimizationParam& param) {
  int loop_kf_start_index = param.history_index;
  int loop_kf_end_index = param.cur_index;

  param.pose_graph_start_index = loop_kf_start_index;
  for (auto iter = history_loop_edgs_.begin(); iter != history_loop_edgs_.end();
       iter++) {
    if ((iter->second.target_kf_index >= loop_kf_start_index &&
         iter->second.target_kf_index <= loop_kf_end_index) ||
        (iter->second.source_kf_index > loop_kf_start_index &&
         iter->second.source_kf_index < loop_kf_end_index)) {
      param.history_loop_closure_edges.push_back(iter->second);
      param.pose_graph_start_index =
          std::min(param.pose_graph_start_index, iter->second.target_kf_index);
    }
  }

  param.pose_graph_start_time =
      cloud_key_pos_xy_->at(param.pose_graph_start_index).timestamp;
  double dt = trajectory_->getDt() / 5.0 * 3.0;
  double tra_max_time = trajectory_->maxTime(LiDARSensor);
  double t = param.pose_graph_start_time;
  bool end_vel = false;
  while (t < tra_max_time || end_vel) {
    SE3d pose = trajectory_->GetIMUPose(t);
    Eigen::Vector3d vel_in_w = trajectory_->GetTransVelWorld(t);

    VelocityData vd;
    vd.timestamp = t;
    vd.velocity = pose.so3().inverse() * vel_in_w;
    vd.gyro = trajectory_->GetRotVelBody(t);

    param.velocity_constraint.push_back(vd);

    if (end_vel) {
      break;
    }

    t += dt;
    if (t > tra_max_time) {
      t = trajectory_->maxTime(LiDARSensor) - trajectory_->getDt() / 5.0;
      end_vel = true;
    }
  }
}

// history_cloud, cur_cloud, T_his_to_G, corrected_T_cur_to_history
void LoopClosure::PublishICPCloud(
    VPointCloud::Ptr history_cloud, VPointCloud::Ptr cur_cloud,
    const Eigen::Matrix4f& T_his_to_G,
    const Eigen::Matrix4f& corrected_T_cur_to_history,
    const Eigen::Matrix4f& guess_T_cur_to_history) {
  VPointCloud target_cloud_in_G;
  sensor_msgs::PointCloud2 target_msg;
  pcl::transformPointCloud(*history_cloud, target_cloud_in_G, T_his_to_G);
  pcl::toROSMsg(target_cloud_in_G, target_msg);
  target_msg.header.frame_id = "/map";

  VPointCloud source_cloud_in_G;
  pcl::transformPointCloud(*cur_cloud, source_cloud_in_G,
                           T_his_to_G * corrected_T_cur_to_history);
  sensor_msgs::PointCloud2 source_msg;
  pcl::toROSMsg(source_cloud_in_G, source_msg);
  source_msg.header.frame_id = "/map";

  VPointCloud cur_cloud_in_his_frame;
  pcl::transformPointCloud(*cur_cloud, cur_cloud_in_his_frame,
                           T_his_to_G * guess_T_cur_to_history);
  sensor_msgs::PointCloud2 check_msg;
  pcl::toROSMsg(cur_cloud_in_his_frame, check_msg);
  check_msg.header.frame_id = "/map";

  pub_icp_target_cloud_.publish(target_msg);
  pub_icp_source_cloud_.publish(source_msg);
  pub_icp_guess_source_cloud_.publish(check_msg);

  {
    static tf::TransformBroadcaster tbr;

    Eigen::Matrix4d T_his_to_G_d = T_his_to_G.cast<double>();
    Eigen::Quaterniond quat(T_his_to_G_d.block<3, 3>(0, 0));
    Eigen::Vector3d pos = T_his_to_G_d.block<3, 1>(0, 3);
    std::string from_frame = "history_lidar";
    std::string to_frame = "map";

    tf::Transform transform;
    transform.setOrigin(tf::Vector3(pos[0], pos[1], pos[2]));
    tf::Quaternion tf_q(quat.x(), quat.y(), quat.z(), quat.w());
    transform.setRotation(tf_q);
    tbr.sendTransform(tf::StampedTransform(transform, ros::Time::now(),
                                           to_frame, from_frame));
  }
}

void LoopClosure::PublishDiscretedPoseGraphMarker(
    const std::vector<SE3d>& pose_before, const std::vector<SE3d>& pose_after) {
  assert(pose_before.size() == pose_after.size() &&
         "[PublishDiscretedPoseGraphMarker] pose size error ");
  visualization_msgs::MarkerArray marker_array;
  // pose before
  visualization_msgs::Marker marker_node_before;
  marker_node_before.header.frame_id = "/map";
  marker_node_before.header.stamp = ros::Time::now();
  marker_node_before.action = visualization_msgs::Marker::ADD;
  marker_node_before.type = visualization_msgs::Marker::SPHERE_LIST;
  marker_node_before.ns = "pose_graph_before";
  marker_node_before.id = 0;
  marker_node_before.pose.orientation.w = 1;
  marker_node_before.scale.x = 0.2;
  marker_node_before.scale.y = 0.2;
  marker_node_before.scale.z = 0.2;
  marker_node_before.color.r = 1.0;
  marker_node_before.color.g = 0;
  marker_node_before.color.b = 0;
  marker_node_before.color.a = 1;

  // pose after
  visualization_msgs::Marker marker_node_after;
  marker_node_after.header.frame_id = "/map";
  marker_node_after.header.stamp = ros::Time::now();
  marker_node_after.action = visualization_msgs::Marker::ADD;
  marker_node_after.type = visualization_msgs::Marker::SPHERE_LIST;
  marker_node_after.ns = "pose_graph_after";
  marker_node_after.id = 0;
  marker_node_after.pose.orientation.w = 1;
  marker_node_after.scale.x = 0.2;
  marker_node_after.scale.y = 0.2;
  marker_node_after.scale.z = 0.2;
  marker_node_after.color.r = 0;
  marker_node_after.color.g = 0;
  marker_node_after.color.b = 1;
  marker_node_after.color.a = 1;

  visualization_msgs::Marker marker_edge;
  marker_edge.header.frame_id = "/map";
  marker_edge.header.stamp = ros::Time::now();
  marker_edge.action = visualization_msgs::Marker::ADD;
  marker_edge.type = visualization_msgs::Marker::LINE_LIST;
  marker_edge.ns = "loop_edges";
  marker_edge.id = 1;
  marker_edge.pose.orientation.w = 1;
  marker_edge.scale.x = 0.02;
  marker_edge.color.r = 0;
  marker_edge.color.g = 0;
  marker_edge.color.b = 1;
  marker_edge.color.a = 1;

  auto const& EP_LtoI = trajectory_->GetSensorEP(LiDARSensor);
  Eigen::Vector3d P_ItoL = EP_LtoI.so3.inverse() * (-EP_LtoI.p);
  for (size_t i = 0; i < pose_after.size(); i++) {
    Eigen::Vector3d p_LinG_bef = pose_before[i].translation();
    Eigen::Vector3d p_IinG_bef = pose_before[i].so3() * P_ItoL + p_LinG_bef;
    geometry_msgs::Point p;
    p.x = p_IinG_bef(0);
    p.y = p_IinG_bef(1);
    p.z = p_IinG_bef(2);
    marker_node_before.points.push_back(p);
    marker_edge.points.push_back(p);

    Eigen::Vector3d p_LinG_aft = pose_after[i].translation();
    Eigen::Vector3d p_IinG_aft = pose_before[i].so3() * P_ItoL + p_LinG_aft;
    p.x = p_IinG_aft(0);
    p.y = p_IinG_aft(1);
    p.z = p_IinG_aft(2);
    marker_node_after.points.push_back(p);
    marker_edge.points.push_back(p);
  }

  marker_array.markers.push_back(marker_node_before);
  marker_array.markers.push_back(marker_node_after);
  marker_array.markers.push_back(marker_edge);

  pub_pose_graph_marker_.publish(marker_array);
}

void LoopClosure::PublishLoopClosureMarkers(const PosCloud::Ptr cloud_key_pos) {
  if (history_loop_info_.empty()) return;
  visualization_msgs::MarkerArray marker_array;
  visualization_msgs::Marker marker_node;
  marker_node.header.frame_id = "/map";
  marker_node.header.stamp = ros::Time::now();
  marker_node.action = visualization_msgs::Marker::ADD;
  marker_node.type = visualization_msgs::Marker::SPHERE_LIST;
  marker_node.ns = "loop_nodes";
  marker_node.id = 0;
  marker_node.pose.orientation.w = 1;
  marker_node.scale.x = 0.3;
  marker_node.scale.y = 0.3;
  marker_node.scale.z = 0.3;
  marker_node.color.r = 0;
  marker_node.color.g = 0.8;
  marker_node.color.b = 1;
  marker_node.color.a = 1;

  // loop edges
  visualization_msgs::Marker marker_edge;
  marker_edge.header.frame_id = "/map";
  marker_edge.header.stamp = ros::Time::now();
  marker_edge.action = visualization_msgs::Marker::ADD;
  marker_edge.type = visualization_msgs::Marker::LINE_LIST;
  marker_edge.ns = "loop_edges";
  marker_edge.id = 1;
  marker_edge.pose.orientation.w = 1;
  marker_edge.scale.x = 0.1;
  marker_edge.color.r = 0.9;
  marker_edge.color.g = 0.9;
  marker_edge.color.b = 0;
  marker_edge.color.a = 1;

  for (auto iter = history_loop_info_.begin(); iter != history_loop_info_.end();
       iter++) {
    int key_cur = iter->first;
    int key_history = iter->second;
    geometry_msgs::Point p;
    const auto& vp_cur = cloud_key_pos->points.at(key_cur);
    p.x = vp_cur.x;
    p.y = vp_cur.y;
    p.z = vp_cur.z;
    marker_node.points.push_back(p);
    marker_edge.points.push_back(p);
    const auto& vp_his = cloud_key_pos->points.at(key_history);
    p.x = vp_his.x;
    p.y = vp_his.y;
    p.z = vp_his.z;
    marker_node.points.push_back(p);
    marker_edge.points.push_back(p);
  }

  marker_array.markers.push_back(marker_node);
  marker_array.markers.push_back(marker_edge);
  pub_loop_closure_marker_.publish(marker_array);
}

}  // namespace clic
