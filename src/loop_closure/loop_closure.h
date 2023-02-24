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
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <lidar_odometry/lidar_odometry.h>
#include <loop_closure/gt_loader.h>

#include <loop_closure/loop_closure_data.h>
#include <loop_closure/pose_graph.h>
#include <utils/cloud_tool.h>
#include <utils/math_utils.h>

namespace clic {

struct LoopClosureParam {
  void Init(const YAML::Node& node) {
    use_full_cloud = node["use_full_cloud"].as<bool>();
    icp_manual_check = node["icp_manual_check"].as<bool>();
    scan_search_radius = node["scan_search_radius"].as<double>();
    scan_search_num = node["scan_search_num"].as<int>();
    scan_time_diff = node["scan_time_diff"].as<double>();
    scan_index_diff = node["scan_index_diff"].as<int>();

    scan_fitness_score = node["scan_fitness_score"].as<double>();

    loop_closure_downsample_leaf_size =
        node["loop_closure_downsample_leaf_size"].as<double>();

    printf("Loop Closure Param:\n");
    printf("\t- scan_search_radius: %.1f\n", scan_search_radius);
    printf("\t- scan_search_num: %d\n", scan_search_num);
    printf("\t- scan_time_diff: %.1f\n", scan_time_diff);
    printf("\t- scan_index_diff: %d\n", scan_index_diff);
    printf("\t- downsample_leaf_size: %.1f\n",
           loop_closure_downsample_leaf_size);
  }

  bool use_full_cloud;
  bool icp_manual_check;

  double scan_search_radius;
  int scan_search_num;
  double scan_time_diff;
  int scan_index_diff;

  double scan_fitness_score;

  double loop_closure_downsample_leaf_size;
};

class LoopClosure {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef std::shared_ptr<LoopClosure> Ptr;

  LoopClosure(const YAML::Node& node, Trajectory::Ptr traj);

  void LoopClosureHandler(LidarOdometry::Ptr lidar_odom_,
                          const GTLoader::Ptr gt_loader);

  bool IsLoopClosureEnable(double cur_time) {
    static double last_check_timestamp = 0;

    if (loop_closure_enable_flag_) {
      double dt1 = cur_time - last_check_timestamp;
      if (dt1 > 10) {
        last_check_timestamp = cur_time;
        if (cur_time - t_last_successed_loop_ >
            (1.0 / loop_closure_frequency_)) {
          return true;
        }
      }
    }

    return false;
  }

  bool HasLoop() const { return has_loop_; }

 private:
 
  bool DetectLoopClosure(PosCloud::Ptr cloud_key_pos_xy_in,
                         std::pair<int, int>& cur_wrt_history);

  bool PerformPoseGraph(std::pair<int, int>& cur_wrt_history,
                        const Eigen::Matrix4f& corrected_T_cur_to_history);

  bool GetLoopPose(const std::pair<int, int>& cur_wrt_history,
                   VPointCloud::Ptr cur_cloud, VPointCloud::Ptr history_cloud,
                   Eigen::Matrix4f& guess_T_cur_to_history,
                   Eigen::Matrix4f& corrected_T_cur_to_history);

  void DownsampleLocalMap(VPointCloud::Ptr& cloud) {
    loop_closure_match_voxel_filter_.setInputCloud(cloud);
    loop_closure_match_voxel_filter_.filter(*cloud);
  }

  bool PointToPointICP(const VPointCloud::Ptr& target_cloud,
                       const VPointCloud::Ptr& source_cloud,
                       double correspondence_distance, Eigen::Matrix4f guess,
                       Eigen::Matrix4f& transform, float& fitness_score);

  void ComputeLoopClosureParam(LoopClosureOptimizationParam& param);

 public:
  void PublishICPCloud(VPointCloud::Ptr history_cloud,
                       VPointCloud::Ptr cur_cloud,
                       const Eigen::Matrix4f& T_his_to_G,
                       const Eigen::Matrix4f& corrected_T_cur_to_history,
                       const Eigen::Matrix4f& guess_T_cur_to_history);

  void PublishDiscretedPoseGraphMarker(const std::vector<SE3d>& pose_before,
                                       const std::vector<SE3d>& pose_after);

  void PublishLoopClosureMarkers(const PosCloud::Ptr cloud_key_pos);

 private:
  bool has_loop_;

  LoopClosureParam lc_param;
  LoopClosureWeights lc_weights;
  Trajectory::Ptr trajectory_;

  PoseGraph pose_graph_node;

  bool loop_closure_enable_flag_;
  double loop_closure_frequency_;
  double t_last_successed_loop_;

  // ICP
  pcl::VoxelGrid<VPoint> loop_closure_match_voxel_filter_;

  PosCloud::Ptr cloud_key_pos_xy_;
  pcl::KdTreeFLANN<PosPoint>::Ptr kdtree_history_key_poses_;

  // loop closure
  std::map<int, int> history_loop_info_;
  std::map<int, RelativePoseData> history_loop_edgs_;

  // pose graph
  ros::NodeHandle nh_;
  ros::Publisher pub_icp_target_cloud_;
  ros::Publisher pub_icp_source_cloud_;
  ros::Publisher pub_icp_guess_source_cloud_;
  ros::Publisher pub_pose_graph_marker_;

  ros::Publisher pub_loop_closure_marker_;
};
}  // namespace clic
