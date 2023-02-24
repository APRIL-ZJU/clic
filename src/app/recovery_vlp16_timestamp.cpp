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

/// read rosbag
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <boost/foreach.hpp>
#define foreach BOOST_FOREACH


#include <utils/mypcl_cloud_type.h>
#include <utils/parameter_struct.h>

#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>  // pcl::fromROSMsg

#include <lidar_odometry/velodyne_feature_extraction.h>
#include <boost/progress.hpp>


inline void RecoverVLP16Timestamp(const VPointCloud::Ptr input_cloud,
                                  RTPointCloud::Ptr output_cloud) {
  // TODO 并不是从上到下依次采集,而是一上一下。
  double VLP16_time_block_[1824][16];
  for (unsigned int w = 0; w < 1824; w++) {
    for (unsigned int h = 0; h < 16; h++) {
      VLP16_time_block_[w][h] =
          h * 2.304 * 1e-6 + w * 55.296 * 1e-6;  /// VLP_16 16*1824
    }
  }

  double lidar_fov_down = -15.0;
  double lidar_fov_resolution = 2.0;

  double first_horizon_angle;
  double max_horizon_angle = 0;
  bool rot_half = false;

  for (size_t i = 0; i < input_cloud->size(); i++) {
    VPoint raw_point = input_cloud->points[i];
    if (!pcl_isfinite(raw_point.x)) continue;
    double depth = sqrt(raw_point.x * raw_point.x + raw_point.y * raw_point.y +
                        raw_point.z * raw_point.z);
    if (depth == 0) continue;
    double pitch = asin(raw_point.z / depth) / M_PI * 180.0;

    int ring = std::round((pitch - lidar_fov_down) / lidar_fov_resolution);
    if (ring < 0 || ring >= 16) continue;

    double horizon_angle = atan2(raw_point.y, -raw_point.x) / M_PI * 180.0;
    if (i == 0) {
      first_horizon_angle = horizon_angle;
    }
    horizon_angle -= first_horizon_angle;
    if (horizon_angle < 0) rot_half = true;
    if (rot_half) horizon_angle += 360;
    int firing = round(horizon_angle / 0.2);
    if (firing < 0 || firing >= 1824) continue;
    double point_time = VLP16_time_block_[firing][ring];

    RTPoint p;
    p.x = raw_point.x;
    p.y = raw_point.y;
    p.z = raw_point.z;

    p.intensity = raw_point.intensity;
    p.ring = ring;
    p.time = (float)point_time;
    output_cloud->push_back(p);

    if (max_horizon_angle < horizon_angle) max_horizon_angle = horizon_angle;
  }

  // static double full_size = 16 * 1824;
  // static double required_size = full_size * 0.8;
  // if (output_cloud->size() < required_size && max_horizon_angle < 350) {
  //   double percent = (double(output_cloud->size()) / full_size);
  //   std::cout << "points percent[" << percent
  //             << "] of /velodyne_points; horizon angle[" << max_horizon_angle
  //             << "]\n";
  // }
}


int main(int argc, char** argv) {
  ros::init(argc, argv, "recovery_vlp16_timestamp");
  ros::NodeHandle nh("~");

  if (argc < 2) {
    std::cerr << "Please input filepath of rosbag. For example \n\n"
              << "rosrun clic recovery_vlp16_timestamp "
                 "/home/user/rosbag/keylab01.bag\n";

    return 0;
  }
  std::string bag_path = argv[1];

  std::string vlp16_points_topic = "/velodyne_points";
  std::string vlp16_points_recovery_topic = vlp16_points_topic + "_recovery";

  rosbag::Bag bag;
  bag.open(bag_path, rosbag::bagmode::Read | rosbag::bagmode::Append);

  std::vector<std::string> topics;
  topics.push_back(vlp16_points_topic);

  rosbag::View view_;
  rosbag::View view_full;
  view_full.addQuery(bag);
  ros::Time time_init = view_full.getBeginTime();
  view_.addQuery(bag, rosbag::TopicQuery(topics), time_init,
                 view_full.getEndTime());

  for (const rosbag::MessageInstance& m : view_) {
    if (m.getTopic() == vlp16_points_topic) {
      auto lidar_msg = m.instantiate<sensor_msgs::PointCloud2>();
      if (clic::VelodyneFeatureExtraction::CheckMsgFields(*lidar_msg)) {
        return 0;
      } else {
        std::cout << "\nTry to recovery timestamp of " << vlp16_points_topic
                  << "\n\n";
        break;
      }
    }
  }

  for (const rosbag::MessageInstance& m : view_) {
    ros::Time rosbag_time = m.getTime();
    // vlp16 points
    if (m.getTopic() == vlp16_points_topic) {
      auto lidar_msg = m.instantiate<sensor_msgs::PointCloud2>();

      /// convert cloud
      VPointCloud::Ptr vlp_raw_cloud(new VPointCloud);
      pcl::fromROSMsg(*lidar_msg, *vlp_raw_cloud);

      // Add time field
      RTPointCloud::Ptr vlp_cloud_with_time(new RTPointCloud);
      RecoverVLP16Timestamp(vlp_raw_cloud, vlp_cloud_with_time);

      sensor_msgs::PointCloud2::Ptr recover_msg(new sensor_msgs::PointCloud2);
      pcl::toROSMsg(*vlp_cloud_with_time, *recover_msg);
      recover_msg->header = lidar_msg->header;
      recover_msg->header.stamp -= ros::Duration(0.1);
      bag.write(vlp16_points_recovery_topic, rosbag_time, recover_msg);
    }
  }
  bag.close();
 
  return 0;
}
