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

#include <eigen_conversions/eigen_msg.h>
#include <nav_msgs/Path.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_broadcaster.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <clic/feature_cloud.h>
#include <clic/imu_array.h>
#include <clic/pose_array.h>

#include <pcl/common/transforms.h>            //pcl::transformPointCloud
#include <pcl_conversions/pcl_conversions.h>  //pcl::fromROSMsg

#include <inertial/imu_state_estimator.h>
#include <lidar_odometry/lidar_feature.h>
#include <spline/trajectory.h>
#include <utils/parameter_struct.h>
#include <visual_odometry/feature_manager.h>

namespace clic {

enum SplineViewerType {
  Init = 0,  //
  // Opt = 1,   //
  Loop = 2,  //
};

class OdometryViewer {
 private:
  // PublishViconData
  ros::Publisher pub_trajectory_raw_;
  ros::Publisher pub_trajectory_est_;

  // PublishIMUData
  ros::Publisher pub_imu_raw_array_;
  ros::Publisher pub_imu_est_array_;

  // PublishLoamCorrespondence
  ros::Publisher pub_target_cloud_;
  ros::Publisher pub_source_cloud_;
  ros::Publisher pub_map_corr_temp_cloud_;

  // PublishDenseCloud
  ros::Publisher pub_target_dense_cloud_;
  ros::Publisher pub_source_dense_cloud_;

  // PublishSplineTrajectory
  ros::Publisher pub_spline_trajectory_;
  ros::Publisher pub_spline_ctrl_;
  ros::Publisher pub_spline_ctrl_cloud_;
  ros::Publisher pub_spline_active_ctrl_cloud_;

  ros::Publisher pub_spline_marg_ctrl_cloud_;

  std::map<SplineViewerType, ros::Publisher> pub_spline_ctrl_type_;
  std::map<SplineViewerType, ros::Publisher> pub_spline_ctrl_cloud_type_;

  /// IMU
  ros::Publisher pub_latest_bias_vio_;
  ros::Publisher pub_latest_bias_lio_;
  // PublishIMUPropogation
  ros::Publisher pub_imu_propogation_trajectory_;

  // PublishImageLandmarks
  ros::Publisher pub_img_landmarks_;
  ros::Publisher pub_img_marg_cloud_;
  ros::Publisher pub_vio_keyframe_;
  ros::Publisher pub_image_track_;

  ros::Publisher pub_odom_gt_;

  // Not used
  ros::Publisher pub_icp_target_cloud_;
  ros::Publisher pub_icp_source_cloud_;
  ros::Publisher pub_icp_raw_source_cloud_;
  ros::Publisher pub_pose_graph_marker_;

 public:
  void SetPublisher(ros::NodeHandle &nh) {
    /// Vicon data
    pub_trajectory_raw_ = nh.advertise<clic::pose_array>("/path_raw", 10);
    pub_trajectory_est_ = nh.advertise<clic::pose_array>("/path_est", 10);
    /// IMU fitting results
    pub_imu_raw_array_ = nh.advertise<clic::imu_array>("/imu_raw_array", 10);
    pub_imu_est_array_ = nh.advertise<clic::imu_array>("/imu_est_array", 10);

    /// ===== LIO ===== ///
    // lidar matching results
    pub_target_cloud_ =
        nh.advertise<sensor_msgs::PointCloud2>("/lio/target_fea", 10);
    pub_source_cloud_ =
        nh.advertise<sensor_msgs::PointCloud2>("/lio/source_fea", 10);
    pub_map_corr_temp_cloud_ =
        nh.advertise<sensor_msgs::PointCloud2>("/lio/map_corr_temp", 10);

    // dense feature cloud
    pub_target_dense_cloud_ =
        nh.advertise<sensor_msgs::PointCloud2>("/lio/target_dense_cloud", 10);
    pub_source_dense_cloud_ =
        nh.advertise<sensor_msgs::PointCloud2>("/lio/source_dense_cloud", 10);

    /// spline trajectory
    pub_spline_trajectory_ =
        nh.advertise<nav_msgs::Path>("/spline/trajectory", 10);
    pub_spline_ctrl_ = nh.advertise<nav_msgs::Path>("/spline/ctrl_path", 10);
    pub_spline_ctrl_cloud_ =
        nh.advertise<sensor_msgs::PointCloud2>("/spline/ctrl_cloud", 10);
    pub_spline_active_ctrl_cloud_ =
        nh.advertise<sensor_msgs::PointCloud2>("/spline/active_ctrl_cloud", 10);
    pub_spline_marg_ctrl_cloud_ =
        nh.advertise<sensor_msgs::PointCloud2>("/spline/marg_ctrl_cloud", 10);

    pub_spline_ctrl_type_[Init] =
        nh.advertise<nav_msgs::Path>("/spline/init_ctrl_path", 10);
    pub_spline_ctrl_cloud_type_[Init] =
        nh.advertise<sensor_msgs::PointCloud2>("/spline/init_ctrl_cloud", 10);

    pub_spline_ctrl_type_[Loop] =
        nh.advertise<nav_msgs::Path>("/spline/loop_ctrl_path", 10);
    pub_spline_ctrl_cloud_type_[Loop] =
        nh.advertise<sensor_msgs::PointCloud2>("/spline/loop_ctrl_cloud", 10);

    pub_latest_bias_vio_ =
        nh.advertise<sensor_msgs::Imu>("/imu/latest_vio_bias", 10);
    pub_latest_bias_lio_ =
        nh.advertise<sensor_msgs::Imu>("/imu/latest_lio_bias", 10);
    /// imu state estimator trajectory
    pub_imu_propogation_trajectory_ =
        nh.advertise<nav_msgs::Path>("/imu/propogation_trajectory", 10);

    /// ===== VIO ===== ///
    pub_img_landmarks_ =
        nh.advertise<sensor_msgs::PointCloud2>("/vio/landmarks", 10);
    pub_img_marg_cloud_ =
        nh.advertise<sensor_msgs::PointCloud2>("/vio/marg_cloud", 10);
    pub_vio_keyframe_ =
        nh.advertise<sensor_msgs::PointCloud2>("/vio/keyframe", 10);
    pub_image_track_ = nh.advertise<sensor_msgs::Image>("/vio/track_image", 10);

    pub_odom_gt_ = nh.advertise<nav_msgs::Path>("/odom_gt", 10);

    std::cout << "[SetPublisher] init done.\n";
  }

  void PublishTF(Eigen::Quaterniond quat, Eigen::Vector3d pos,
                 std::string from_frame, std::string to_frame) {
    static tf::TransformBroadcaster tbr;
    tf::Transform transform;
    transform.setOrigin(tf::Vector3(pos[0], pos[1], pos[2]));
    tf::Quaternion tf_q(quat.x(), quat.y(), quat.z(), quat.w());
    transform.setRotation(tf_q);
    tbr.sendTransform(tf::StampedTransform(transform, ros::Time::now(),
                                           to_frame, from_frame));
  }

  void PublishIMUData(Trajectory::Ptr trajectory,
                      const Eigen::aligned_vector<IMUData> &imu_data,
                      const IMUBias &bias, const Eigen::Vector3d &gravity) {
    clic::imu_array imu_array_raw;
    clic::imu_array imu_array_est;

    for (auto const &v : imu_data) {
      geometry_msgs::Vector3 gyro, accel;
      tf::vectorEigenToMsg(v.gyro, gyro);
      tf::vectorEigenToMsg(v.accel, accel);
      imu_array_raw.timestamps.push_back(v.timestamp);
      imu_array_raw.angular_velocities.push_back(gyro);
      imu_array_raw.linear_accelerations.push_back(accel);

      Eigen::Vector3d w_b =
          trajectory->GetRotVelBody(v.timestamp) + bias.gyro_bias;
      Eigen::Vector3d a_w = trajectory->GetTransAccelWorld(v.timestamp);
      SE3d pose = trajectory->GetIMUPose(v.timestamp);
      Eigen::Vector3d a_b =
          pose.so3().inverse() * (a_w + gravity) + bias.accel_bias;

      /// some special cases for calibrating between Arpose and IMU
      if (v.gyro.norm() < 1e-10) w_b = Eigen::Vector3d::Zero();
      if (v.accel.norm() < 1e-10) a_b = Eigen::Vector3d::Zero();

      geometry_msgs::Vector3 gyro2, accel2;
      tf::vectorEigenToMsg(w_b, gyro2);
      tf::vectorEigenToMsg(a_b, accel2);
      imu_array_est.timestamps.push_back(v.timestamp);
      imu_array_est.angular_velocities.push_back(gyro2);
      imu_array_est.linear_accelerations.push_back(accel2);
    }
    imu_array_raw.header.stamp = ros::Time::now();
    imu_array_raw.header.frame_id = "/imu";

    imu_array_est.header = imu_array_raw.header;

    pub_imu_raw_array_.publish(imu_array_raw);
    pub_imu_est_array_.publish(imu_array_est);
  }

  void PublishViconData(Trajectory::Ptr trajectory,
                        Eigen::aligned_vector<PoseData> &vicon_data) {
    clic::pose_array vicon_path_raw;
    clic::pose_array vicon_path_est;

    for (auto const &v : vicon_data) {
      geometry_msgs::Vector3 position;
      geometry_msgs::Quaternion orientation;

      // raw data
      tf::vectorEigenToMsg(v.position, position);
      tf::quaternionEigenToMsg(v.orientation.unit_quaternion(), orientation);

      vicon_path_raw.timestamps.push_back(v.timestamp);
      vicon_path_raw.positions.push_back(position);
      vicon_path_raw.orientations.push_back(orientation);

      // estiamted pose
      SE3d pose = trajectory->GetLidarPose(v.timestamp);
      tf::vectorEigenToMsg(pose.translation(), position);
      tf::quaternionEigenToMsg(pose.unit_quaternion(), orientation);
      vicon_path_est.timestamps.push_back(v.timestamp);
      vicon_path_est.positions.push_back(position);
      vicon_path_est.orientations.push_back(orientation);
    }

    vicon_path_raw.header.frame_id = "/map";
    vicon_path_raw.header.stamp = ros::Time::now();
    vicon_path_est.header = vicon_path_raw.header;

    pub_trajectory_raw_.publish(vicon_path_raw);
    pub_trajectory_est_.publish(vicon_path_est);
  }

  void PublishLoamMapCorrTemp(Trajectory::Ptr trajectory,
                              const VPointCloud &map_corr_temp_cloud) {
    if (pub_map_corr_temp_cloud_.getNumSubscribers() == 0) return;

    sensor_msgs::PointCloud2 target_msg;
    pcl::toROSMsg(map_corr_temp_cloud, target_msg);
    target_msg.header.stamp = ros::Time::now();
    target_msg.header.frame_id = "/map";

    pub_map_corr_temp_cloud_.publish(target_msg);
  }

  void PublishLoamCorrespondence(
      Trajectory::Ptr trajectory,
      const Eigen::aligned_vector<PointCorrespondence> &point_measurement) {
    if (pub_source_cloud_.getNumSubscribers() == 0 &&
        pub_target_cloud_.getNumSubscribers() == 0)
      return;

    VPointCloud target_cloud, source_cloud;
    for (const PointCorrespondence &cor : point_measurement) {
      if (cor.t_map < trajectory->minTime(LiDARSensor) ||
          cor.t_point < trajectory->minTime(LiDARSensor)) {
        LOG(INFO) << "[PublishLoamCorrespondence] skip " << cor.t_map << "; "
                  << cor.t_point;
        continue;
      }

      SE3d T_LktoG = trajectory->GetLidarPose(cor.t_point);
      // SE3d T_MtoG = trajectory->GetLidarPose(cor.t_map);
      // Eigen::Vector3d p_inM = T_MtoG.inverse() * T_LktoG * cor.point;
      Eigen::Vector3d p_inM = T_LktoG * cor.point;

      VPoint p_target, p_source;
      Eigen::Vector3d p_intersect;
      if (cor.geo_type == GeometryType::Plane) {
        double dist = p_inM.dot(cor.geo_plane.head(3)) + cor.geo_plane[3];
        p_intersect = p_inM + dist * cor.geo_plane.head(3);

        p_target.intensity = 100;
        p_source.intensity = 100;
      } else {
        double t = (p_inM - cor.geo_point).dot(cor.geo_normal);
        p_intersect = cor.geo_point - t * cor.geo_normal;

        p_target.intensity = 50;
        p_source.intensity = 50;
      }

      p_target.x = p_intersect[0];
      p_target.y = p_intersect[1];
      p_target.z = p_intersect[2];
      target_cloud.push_back(p_target);
      p_source.x = p_inM[0];
      p_source.y = p_inM[1];
      p_source.z = p_inM[2];
      source_cloud.push_back(p_source);
    }

    sensor_msgs::PointCloud2 target_msg, source_msg;
    pcl::toROSMsg(target_cloud, target_msg);
    pcl::toROSMsg(source_cloud, source_msg);

    target_msg.header.stamp = ros::Time::now();
    target_msg.header.frame_id = "/map";
    source_msg.header = target_msg.header;

    pub_target_cloud_.publish(target_msg);
    pub_source_cloud_.publish(source_msg);
  }

  void PublishSplineType(Trajectory::Ptr trajectory,
                         SplineViewerType viewer_type) {
    double opt_min_time = 0;
    int opt_fixed_idx = -1;
    if (viewer_type == Init) {
      opt_min_time = trajectory->opt_min_init_time_tmp;
      opt_fixed_idx = trajectory->opt_init_fixed_idx_tmp;
    } else if (viewer_type == Loop) {
      opt_min_time = trajectory->opt_min_loop_time_tmp;
      opt_fixed_idx = trajectory->opt_loop_fixed_idx_tmp;
    }
    size_t start_idx = trajectory->GetCtrlIndex(opt_min_time * S_TO_NS);
    if (opt_fixed_idx < 0) opt_fixed_idx = 0;

    ros::Time time_now = ros::Time::now();
    ros::Time t_temp;
    if (pub_spline_ctrl_type_[viewer_type].getNumSubscribers() != 0) {
      double min_time = trajectory->minTime(LiDARSensor);
      if (min_time < 0) min_time = 0;

      std::vector<geometry_msgs::PoseStamped> poses_ctrl;
      for (size_t i = start_idx; i < trajectory->numKnots(); ++i) {
        double t = min_time + i * trajectory->getDt();
        geometry_msgs::PoseStamped geo_ctrl;
        geo_ctrl.header.stamp = t_temp.fromSec(t);
        geo_ctrl.header.frame_id = "/map";
        tf::pointEigenToMsg(trajectory->getKnotPos(i), geo_ctrl.pose.position);
        tf::quaternionEigenToMsg(trajectory->getKnotSO3(i).unit_quaternion(),
                                 geo_ctrl.pose.orientation);
        poses_ctrl.push_back(geo_ctrl);
      }

      nav_msgs::Path traj_ctrl;
      traj_ctrl.header.stamp = time_now;
      traj_ctrl.header.frame_id = "/map";
      traj_ctrl.poses = poses_ctrl;
      pub_spline_ctrl_type_[viewer_type].publish(traj_ctrl);
    }

    if (pub_spline_ctrl_cloud_type_[viewer_type].getNumSubscribers() != 0) {
      VPointCloud ctrl_cloud;
      for (size_t i = start_idx; i < trajectory->numKnots(); ++i) {
        const Eigen::Vector3d &p = trajectory->getKnotPos(i);
        VPoint ctrl_p;
        ctrl_p.x = p[0];
        ctrl_p.y = p[1];
        ctrl_p.z = p[2];
        if ((int)i <= opt_fixed_idx)
          ctrl_p.intensity = 100;
        else
          ctrl_p.intensity = 200;

        ctrl_cloud.push_back(ctrl_p);
      }

      sensor_msgs::PointCloud2 cloud_msg;
      pcl::toROSMsg(ctrl_cloud, cloud_msg);
      cloud_msg.header.stamp = ros::Time::now();
      cloud_msg.header.frame_id = "/map";
      pub_spline_ctrl_cloud_type_[viewer_type].publish(cloud_msg);
    }
  }

  void PublishMargCtrlCloud(const VPointCloud &marg_ctrl_cloud) {
    if (pub_spline_marg_ctrl_cloud_.getNumSubscribers() != 0 &&
        marg_ctrl_cloud.size() > 0) {
      sensor_msgs::PointCloud2 cloud_msg;
      pcl::toROSMsg(marg_ctrl_cloud, cloud_msg);
      cloud_msg.header.stamp = ros::Time::now();
      cloud_msg.header.frame_id = "/map";
      pub_spline_marg_ctrl_cloud_.publish(cloud_msg);
    }
  }

  void PublishSplineTrajectory(Trajectory::Ptr trajectory, double min_time,
                               double max_time, double dt) {
    ros::Time time_now = ros::Time::now();
    ros::Time t_temp;
    if (min_time < trajectory->minTime(IMUSensor))
      min_time = trajectory->minTime(IMUSensor);
    if (max_time > trajectory->maxTime(IMUSensor))
      max_time = trajectory->maxTime(IMUSensor);

    if (min_time < 0) min_time = 0;
    if (max_time <= min_time) return;

    if (pub_spline_trajectory_.getNumSubscribers() != 0) {
      std::vector<geometry_msgs::PoseStamped> poses_geo;
      for (double t = min_time; t < max_time; t += dt) {
        SE3d pose = trajectory->GetIMUPose(t);
        geometry_msgs::PoseStamped poseIinG;
        poseIinG.header.stamp = t_temp.fromSec(t);
        poseIinG.header.frame_id = "/map";
        tf::pointEigenToMsg(pose.translation(), poseIinG.pose.position);
        tf::quaternionEigenToMsg(pose.unit_quaternion(),
                                 poseIinG.pose.orientation);
        poses_geo.push_back(poseIinG);
      }

      nav_msgs::Path traj_path;
      traj_path.header.stamp = time_now;
      traj_path.header.frame_id = "/map";
      traj_path.poses = poses_geo;
      pub_spline_trajectory_.publish(traj_path);
    }

    if (pub_spline_ctrl_.getNumSubscribers() != 0) {
      std::vector<geometry_msgs::PoseStamped> poses_ctrl;
      for (size_t i = 0; i < trajectory->numKnots(); ++i) {
        double t = min_time + i * trajectory->getDt();
        geometry_msgs::PoseStamped geo_ctrl;
        geo_ctrl.header.stamp = t_temp.fromSec(t);
        geo_ctrl.header.frame_id = "/map";
        tf::pointEigenToMsg(trajectory->getKnotPos(i), geo_ctrl.pose.position);
        tf::quaternionEigenToMsg(trajectory->getKnotSO3(i).unit_quaternion(),
                                 geo_ctrl.pose.orientation);
        poses_ctrl.push_back(geo_ctrl);
      }

      nav_msgs::Path traj_ctrl;
      traj_ctrl.header.stamp = time_now;
      traj_ctrl.header.frame_id = "/map";
      traj_ctrl.poses = poses_ctrl;
      pub_spline_ctrl_.publish(traj_ctrl);
    }

    if (pub_spline_ctrl_cloud_.getNumSubscribers() != 0) {
      VPointCloud ctrl_cloud;
      for (size_t i = 0; i < trajectory->numKnots(); ++i) {
        const Eigen::Vector3d &p = trajectory->getKnotPos(i);
        VPoint ctrl_p;
        ctrl_p.x = p[0];
        ctrl_p.y = p[1];
        ctrl_p.z = p[2];
        ctrl_p.intensity = 100;

        ctrl_cloud.push_back(ctrl_p);
      }

      sensor_msgs::PointCloud2 cloud_msg;
      pcl::toROSMsg(ctrl_cloud, cloud_msg);
      cloud_msg.header.stamp = ros::Time::now();
      cloud_msg.header.frame_id = "/map";
      pub_spline_ctrl_cloud_.publish(cloud_msg);
    }

    if (pub_spline_active_ctrl_cloud_.getNumSubscribers() != 0) {
      size_t opt_idx =
          trajectory->GetCtrlIndex(trajectory->opt_min_lio_time_tmp * S_TO_NS);

      VPointCloud active_ctrl_cloud;
      for (size_t i = opt_idx; i < trajectory->numKnots(); ++i) {
        const Eigen::Vector3d &p = trajectory->getKnotPos(i);
        VPoint ctrl_p;
        ctrl_p.x = p[0];
        ctrl_p.y = p[1];
        ctrl_p.z = p[2];
        ctrl_p.intensity = 100;

        active_ctrl_cloud.push_back(ctrl_p);
      }

      sensor_msgs::PointCloud2 active_cloud_msg;
      pcl::toROSMsg(active_ctrl_cloud, active_cloud_msg);
      active_cloud_msg.header.stamp = ros::Time::now();
      active_cloud_msg.header.frame_id = "/map";
      pub_spline_active_ctrl_cloud_.publish(active_cloud_msg);
    }
  }

  void PublishLatestBias(const Eigen::Vector3d &gyro_bias,
                         const Eigen::Vector3d &accel_bias,
                         bool is_LIO = true) {
    if (pub_latest_bias_vio_.getNumSubscribers() != 0 ||
        pub_latest_bias_lio_.getNumSubscribers() != 0) {
      sensor_msgs::Imu imu_msg;
      imu_msg.header.stamp = ros::Time::now();
      imu_msg.angular_velocity.x = gyro_bias[0];
      imu_msg.angular_velocity.y = gyro_bias[1];
      imu_msg.angular_velocity.z = gyro_bias[2];
      imu_msg.linear_acceleration.x = accel_bias[0];
      imu_msg.linear_acceleration.y = accel_bias[1];
      imu_msg.linear_acceleration.z = accel_bias[2];
      if (is_LIO)
        pub_latest_bias_lio_.publish(imu_msg);
      else
        pub_latest_bias_vio_.publish(imu_msg);
    }
  }

  void PublishOdomGt(const std::vector<PoseData> &poses_gt, double max_time) {
    if (pub_odom_gt_.getNumSubscribers() == 0 || poses_gt.empty()) return;

    ros::Time time_now = ros::Time::now();
    ros::Time t_temp;
    std::vector<geometry_msgs::PoseStamped> poses_geo;
    for (auto const &v : poses_gt) {
      if (v.timestamp > max_time) break;

      geometry_msgs::PoseStamped pose_gt;
      pose_gt.header.stamp = t_temp.fromSec(v.timestamp);
      pose_gt.header.frame_id = "/map";
      tf::pointEigenToMsg(v.position, pose_gt.pose.position);
      tf::quaternionEigenToMsg(v.orientation.unit_quaternion(),
                               pose_gt.pose.orientation);
      poses_geo.push_back(pose_gt);
    }

    nav_msgs::Path traj_path;
    traj_path.header.stamp = time_now;
    traj_path.header.frame_id = "/map";
    traj_path.poses = poses_geo;

    pub_odom_gt_.publish(traj_path);
  }

  geometry_msgs::PoseStamped fromIMUState(const IMUState &imu_state) const {
    ros::Time t_temp;
    geometry_msgs::PoseStamped poseIinG;
    poseIinG.header.stamp = t_temp.fromSec(imu_state.timestamp);
    poseIinG.header.frame_id = "/map";
    tf::pointEigenToMsg(imu_state.p, poseIinG.pose.position);
    tf::quaternionEigenToMsg(imu_state.q, poseIinG.pose.orientation);

    return poseIinG;
  }

  void PublishIMUEstimator(const ImuStateEstimator::Ptr &imu_estimator) {
    /// Propogation
    std::vector<geometry_msgs::PoseStamped> poses_propogation;
    poses_propogation.push_back(
        fromIMUState(imu_estimator->GetPropagateStartState()));
    poses_propogation.push_back(fromIMUState(imu_estimator->GetIMUState()));

    nav_msgs::Path traj_propogation;
    traj_propogation.header.stamp = ros::Time::now();
    traj_propogation.header.frame_id = "/map";
    traj_propogation.poses = poses_propogation;

    pub_imu_propogation_trajectory_.publish(traj_propogation);
  }

  void PublishDenseCloud(Trajectory::Ptr trajectory,
                         const LiDARFeature &target_feature,
                         const LiDARFeature &source_feature) {
    static int skip = 0;
    if (pub_target_dense_cloud_.getNumSubscribers() != 0 &&
        (++skip % 20 == 0)) {
      PosCloud target_cloud;
      double target_timestamp = target_feature.timestamp;
      SE3d target_pos = trajectory->GetLidarPose(target_timestamp);
      Eigen::Matrix4d tranform_matrix = target_pos.matrix();

      tranform_matrix = Eigen::Matrix4d::Identity();
      pcl::transformPointCloud(*target_feature.surface_features, target_cloud,
                               tranform_matrix);

      sensor_msgs::PointCloud2 target_msg;
      pcl::toROSMsg(target_cloud, target_msg);
      target_msg.header.stamp = ros::Time::now();
      target_msg.header.frame_id = "/map";
      pub_target_dense_cloud_.publish(target_msg);
    }

    if (pub_source_dense_cloud_.getNumSubscribers() != 0) {
      VPointCloud source_cloud;
      for (size_t i = 0; i < source_feature.surface_features->size(); i++) {
        double point_timestamp =
            source_feature.surface_features->points[i].timestamp;
        SE3d point_pos = trajectory->GetLidarPose(point_timestamp);
        Eigen::Vector3d point_local(
            source_feature.surface_features->points[i].x,
            source_feature.surface_features->points[i].y,
            source_feature.surface_features->points[i].z);
        Eigen::Vector3d point_out =
            point_pos.so3() * point_local + point_pos.translation();
        VPoint p;
        p.x = point_out(0);
        p.y = point_out(1);
        p.z = point_out(2);
        p.intensity = source_feature.surface_features->points[i].intensity;
        source_cloud.push_back(p);
      }

      sensor_msgs::PointCloud2 source_msg;
      pcl::toROSMsg(source_cloud, source_msg);
      source_msg.header.stamp = ros::Time::now();
      source_msg.header.frame_id = "/map";
      pub_source_dense_cloud_.publish(source_msg);
    }
  }

  void PublishImageLandmarks(const VPointCloud &landmarks,
                             const VPointCloud &marg_cloud) {
    if (pub_img_landmarks_.getNumSubscribers() != 0 && !landmarks.empty()) {
      sensor_msgs::PointCloud2 cloud_msg;
      pcl::toROSMsg(landmarks, cloud_msg);
      cloud_msg.header.stamp = ros::Time::now();
      cloud_msg.header.frame_id = "/map";

      pub_img_landmarks_.publish(cloud_msg);
    }

    if (pub_img_marg_cloud_.getNumSubscribers() != 0 && !marg_cloud.empty()) {
      sensor_msgs::PointCloud2 cloud_msg;
      pcl::toROSMsg(marg_cloud, cloud_msg);
      cloud_msg.header.stamp = ros::Time::now();
      cloud_msg.header.frame_id = "/map";

      pub_img_marg_cloud_.publish(cloud_msg);
    }
  }

  void PublishTrackImage(const std::list<FeaturePerId> &features,
                         const std::vector<int> &good_feature_ids,
                         cv_bridge::CvImageConstPtr img_ptr) {
    if (pub_image_track_.getNumSubscribers() == 0) return;

    img_ptr = cv_bridge::cvtColor(img_ptr, sensor_msgs::image_encodings::BGR8);
    cv::Mat image = img_ptr->image;

    int r1 = 255, g1 = 0, b1 = 255;
    int r2 = 255, g2 = 255, b2 = 255;

    for (const auto &it_per_id : features) {
      if (std::find(good_feature_ids.begin(), good_feature_ids.end(),
                    it_per_id.feature_id) == good_feature_ids.end())
        continue;

      int win_len = it_per_id.feature_per_frame.size() - 1;
      for (int idx = win_len; idx > 0; --idx) {
        int r = r2 - r1 / win_len * idx;
        int g = g2 - g1 / win_len * idx;
        int b = b2 - b1 / win_len * idx;

        // Draw current point
        const auto &uv = it_per_id.feature_per_frame[idx].uv;
        cv::Point2f pt_c(uv(0), uv(1));
        cv::circle(image, pt_c, 2, cv::Scalar(r, g, b), cv::FILLED);
        // Display the line from this point to the next
        if (idx + 1 < win_len) {
          const auto &uv = it_per_id.feature_per_frame[idx + 1].uv;
          cv::Point2f pt_n(uv(0), uv(1));
          cv::line(image, pt_c, pt_n, cv::Scalar(r, g, b), 3);
        }
      }
    }
    pub_image_track_.publish(img_ptr->toImageMsg());
  }

  void PublishVioKeyFrame(Trajectory::Ptr trajectory,
                          const double timestamps[]) {
    if (pub_vio_keyframe_.getNumSubscribers() == 0) return;

    VPointCloud keyframe_cloud;
    for (size_t i = 0; i <= WINDOW_SIZE; i++) {
      auto pos = trajectory->GetCameraPose(timestamps[i]).translation();
      VPoint p;
      p.x = pos.x();
      p.y = pos.y();
      p.z = pos.z();
      p.intensity = 200;
      keyframe_cloud.push_back(p);
    }

    sensor_msgs::PointCloud2 cloud_msg;
    pcl::toROSMsg(keyframe_cloud, cloud_msg);
    cloud_msg.header.stamp = ros::Time::now();
    cloud_msg.header.frame_id = "/map";

    pub_vio_keyframe_.publish(cloud_msg);
  }

  void ErrorStatistics(
      Trajectory::Ptr trajectory,
      const Eigen::aligned_vector<PointCorrespondence> &point_measurement,
      const Eigen::aligned_vector<IMUData> &imu_measurement,
      std::string note = " ") {
    double corner_error = 0;
    double surface_error = 0;
    int corner_cnt = 0;
    int surface_cnt = 0;
    for (const PointCorrespondence &cor : point_measurement) {
      if (cor.t_map < trajectory->minTime(LiDARSensor) ||
          cor.t_point < trajectory->minTime(LiDARSensor)) {
        LOG(INFO) << "[PublishLoamCorrespondence] skip " << cor.t_map << "; "
                  << cor.t_point;
        continue;
      }

      SE3d T_MtoG = trajectory->GetLidarPose(cor.t_map);
      SE3d T_LktoG = trajectory->GetLidarPose(cor.t_point);
      Eigen::Vector3d p_inM = T_MtoG.inverse() * T_LktoG * cor.point;

      if (cor.geo_type == GeometryType::Plane) {
        double dist = p_inM.dot(cor.geo_plane.head(3)) + cor.geo_plane[3];
        surface_error += fabs(dist);
        surface_cnt++;
      } else {
        double dist = (p_inM - cor.geo_point).cross(cor.geo_normal).norm();
        corner_error += dist;
        corner_cnt++;
      }
    }

    LOG(INFO) << "[Corner] Error : " << corner_error / corner_cnt;
    LOG(INFO) << "[Surface] Error : " << surface_error / surface_cnt;
  }
};
}  // namespace clic
