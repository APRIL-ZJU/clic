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

/// read rosbag
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <boost/foreach.hpp>
#define foreach BOOST_FOREACH
#include <livox_ros_driver/CustomMsg.h>       /// livox
#include <pcl_conversions/pcl_conversions.h>  // pcl::fromROSMsg
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>

#include <Eigen/Dense>
#include <algorithm>
#include <map>
#include <vector>

#include <lidar_odometry/livox_feature_extraction.h>
#include <lidar_odometry/velodyne_feature_extraction.h>
#include <visual_odometry/visual_feature/feature_tracker_node.h>

#include <utils/log_utils.h>
#include <utils/parameter_struct.h>
#include <utils/eigen_utils.hpp>

namespace clic {

enum OdometryMode {
  LIO = 0,  //
  LICO = 1,
};

enum LiDARType {
  VLP = 0,
  LIVOX,
};

struct NextMsgs {
  NextMsgs()
      : lidar_timestamp(-1),
        lidar_max_timestamp(-1),
        lidar_raw_cloud(new RTPointCloud),
        lidar_surf_cloud(new RTPointCloud),
        lidar_corner_cloud(new RTPointCloud) {}

  void Clear() {
    scan_num = 0;
    lidar_timestamp = -1;
    lidar_max_timestamp = -1;
    lidar_raw_cloud->clear();
    lidar_surf_cloud->clear();
    lidar_corner_cloud->clear();

    image_feature_msgs.clear();
  }

  void CheckData() {
    double max_time[3];
    max_time[0] = pcl::GetCloudMaxTime(lidar_surf_cloud);
    max_time[1] = pcl::GetCloudMaxTime(lidar_corner_cloud);
    max_time[2] = pcl::GetCloudMaxTime(lidar_raw_cloud);
    for (int i = 0; i < 3; i++) {
      if ((max_time[i] - lidar_max_timestamp) > 1e-6)
        std::cout << YELLOW << "[CheckData] Problem !! " << i
                  << " desired max time: " << lidar_max_timestamp
                  << "; computed max_time: " << max_time[i] << "\n"
                  << RESET;
    }
  }

  int scan_num = 0;
  double lidar_timestamp;      // time wrt. start of the traj
  double lidar_max_timestamp;  // time wrt. start of the traj
  RTPointCloud::Ptr lidar_raw_cloud;
  RTPointCloud::Ptr lidar_surf_cloud;
  RTPointCloud::Ptr lidar_corner_cloud;

  // image features
  std::vector<sensor_msgs::PointCloud::ConstPtr> image_feature_msgs;
};

struct LiDARCloudData {
  LiDARCloudData()
      : lidar_id(0),
        timestamp(0),
        max_timestamp(0),
        raw_cloud(new RTPointCloud),
        surf_cloud(new RTPointCloud),
        corner_cloud(new RTPointCloud),
        is_time_wrt_traj_start(false) {}
  int lidar_id;
  double timestamp;
  // max time of points wrt. start of the traj
  double max_timestamp;

  RTPointCloud::Ptr raw_cloud;
  RTPointCloud::Ptr surf_cloud;
  RTPointCloud::Ptr corner_cloud;

  bool is_time_wrt_traj_start;

  void ToRelativeMeasureTime(double traj_start_time) {

    CloudToRelativeMeasureTime(raw_cloud, timestamp, traj_start_time);
    CloudToRelativeMeasureTime(surf_cloud, timestamp, traj_start_time);
    CloudToRelativeMeasureTime(corner_cloud, timestamp, traj_start_time);
    timestamp -= traj_start_time;

    max_timestamp = pcl::GetCloudMaxTime(raw_cloud);

    double surf_max = pcl::GetCloudMaxTime(surf_cloud);
    if (surf_max > max_timestamp) {
      std::cout << RED << "surf/raw cloud max time: " << surf_max << "/"
                << max_timestamp << RESET << std::endl;
      max_timestamp = surf_max;
    }

    is_time_wrt_traj_start = true;
  }
};

class MsgManager {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef std::shared_ptr<MsgManager> Ptr;

  MsgManager(const YAML::Node& node, ros::NodeHandle& nh);

  // read msgs from rosbag
  void SpinBagOnce();

  bool GetNextMsgs(double traj_max, double start_time, double knot_dt,
                   bool in_scan_unit = true);

  double GetCurIMUTimestamp() const { return cur_imu_timestamp_; }

  // show the cache msg info 
  void LogInfo() const;

  int NumLiDAR() const { return num_lidars_; }

  // get the latest image for visualization
  cv_bridge::CvImageConstPtr GetLatestImage() {
    return feature_tracker_node_->GetLatestImage();
  }

  static void IMUMsgToIMUData(const sensor_msgs::Imu::ConstPtr& imu_msg,
                              IMUData& data) {
    data.timestamp = imu_msg->header.stamp.toSec();
    data.gyro = Eigen::Vector3d(imu_msg->angular_velocity.x,
                                imu_msg->angular_velocity.y,
                                imu_msg->angular_velocity.z);
    data.accel = Eigen::Vector3d(imu_msg->linear_acceleration.x,
                                 imu_msg->linear_acceleration.y,
                                 imu_msg->linear_acceleration.z);
    Eigen::Vector4d q(imu_msg->orientation.w, imu_msg->orientation.x,
                      imu_msg->orientation.y, imu_msg->orientation.z);
    if (std::fabs(q.norm() - 1) < 0.01) {
      data.orientation = SO3d(Eigen::Quaterniond(q[0], q[1], q[2], q[3]));
    }
  }

 private:
  void LoadBag(const YAML::Node& node);

  // remove the msgs before the start_time
  void RemoveBeginData(double start_time, double relative_start_time = 0);

  bool HasEnvMsg() const;

  // check if we have enough data to estimate the traj
  bool CheckMsgIsReady(double traj_max, double start_time, double knot_dt,
                       bool in_scan_unit) const;

  // get the cloud msg to be processed
  bool AddToNextMsg(std::deque<LiDARCloudData>::iterator scan, double traj_max,
                    bool in_scan_unit);

  void IMUMsgHandle(const sensor_msgs::Imu::ConstPtr& imu_msg);

  // in offline mode, feed image to feature_tracker_node
  void ImageMsgHandle(const sensor_msgs::Image::ConstPtr& image_msg) {
    feature_tracker_node_->img_callback(image_msg);
  }

  void VelodyneMsgHandle(const sensor_msgs::PointCloud2::ConstPtr& vlp16_msg,
                         int lidar_id);

  void LivoxMsgHandle(const livox_ros_driver::CustomMsg::ConstPtr& livox_msg,
                      int lidar_id);

 public:
  bool has_valid_msg_;

  std::string bag_path_;

  NextMsgs next_msgs;
  Eigen::aligned_deque<IMUData> imu_buf_;  // clic trajectory estimator

  double t_offset_imu_;
  double t_offset_camera_;
  
  // only for ntu dataset
  double t_offset_lidar_;

  double imu_period_s_;
  double add_extra_timeoffset_s_;

  double t_image_ms_;
  double t_lidar_ms_;

 private:
  double cur_imu_timestamp_;
  std::deque<LiDARCloudData> lidar_buf_;
  std::vector<double> lidar_max_timestamps_;

  bool use_image_;

  std::string imu_topic_;
  int num_lidars_;
  std::vector<LiDARType> lidar_types;
  std::vector<std::string> lidar_topics_;
  std::string image_topic_;

  std::vector<ExtrinsicParam> EP_LktoI_;

  /// extrinsics form lidar_k to lidar_0
  Eigen::aligned_vector<Eigen::Matrix4d> T_LktoL0_vec_;

  rosbag::Bag bag_;
  rosbag::View view_;

  ros::Subscriber sub_imu_;
  std::vector<ros::Subscriber> subs_vlp16_;
  std::vector<ros::Subscriber> subs_livox_;
  ros::Subscriber sub_image_;

  feature_tracker::FeatureTrackerNode::Ptr feature_tracker_node_;

  VelodyneFeatureExtraction::Ptr velodyne_feature_extraction_;
  LivoxFeatureExtraction::Ptr livox_feature_extraction_;
};

}  // namespace clic
