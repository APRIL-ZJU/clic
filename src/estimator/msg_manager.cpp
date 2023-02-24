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

#include <estimator/msg_manager.h>
#include <utils/parameter_struct.h>

#include <pcl/common/transforms.h>

namespace clic {

MsgManager::MsgManager(const YAML::Node& node, ros::NodeHandle& nh)
    : has_valid_msg_(true),
      t_offset_imu_(0),
      t_offset_camera_(0),
      t_offset_lidar_(0),
      cur_imu_timestamp_(-1),
      use_image_(false),
      image_topic_("") {
  std::string config_path = node["config_path"].as<std::string>();

  OdometryMode odom_mode = OdometryMode(node["odometry_mode"].as<int>());
  MODE mode = MODE(node["mode"].as<int>());  // MODE::Odometry_Offline

  nh.param<std::string>("bag_path", bag_path_, "");
  if (bag_path_ == "") {
    bag_path_ = node["bag_path"].as<std::string>();
  }

  /// IMU topic
  std::string imu_yaml = node["imu_yaml"].as<std::string>();
  YAML::Node imu_node = YAML::LoadFile(config_path + imu_yaml);
  imu_topic_ = imu_node["imu_topic"].as<std::string>();

  double imu_frequency = node["imu_frequency"].as<double>();
  imu_period_s_ = 1. / imu_frequency;

  add_extra_timeoffset_s_ =
      yaml::GetValue<double>(node, "add_extra_timeoffset_s", 0);
  LOG(INFO) << "add_extra_timeoffset_s: " << add_extra_timeoffset_s_;
  std::cout << "add_extra_timeoffset_s: " << add_extra_timeoffset_s_ << "\n";
  std::cout << "imu_period_s : " << imu_period_s_ << "\n";

  /// image topic
  if (OdometryMode::LICO == odom_mode) use_image_ = true;
  if (use_image_) {
    std::string cam_yaml = config_path + node["camera_yaml"].as<std::string>();
    YAML::Node cam_node = YAML::LoadFile(cam_yaml);
    image_topic_ = cam_node["image_topic"].as<std::string>();

    bool fea_track_offline = (mode == MODE::Odometry_Offline) ? true : false;
    feature_tracker_node_ =
        std::make_shared<feature_tracker::FeatureTrackerNode>(
            cam_yaml, fea_track_offline, add_extra_timeoffset_s_);
  }

  /// lidar topic
  std::string lidar_yaml = node["lidar_odom_yaml"].as<std::string>();
  YAML::Node lidar_node = YAML::LoadFile(config_path + lidar_yaml);
  num_lidars_ = lidar_node["num_lidars"].as<int>();

  bool use_livox = false;
  bool use_vlp = false;
  for (int i = 0; i < num_lidars_; ++i) {
    std::string lidar_str = "lidar" + std::to_string(i);
    const auto& lidar_i = lidar_node[lidar_str];
    bool is_livox = lidar_i["is_livox"].as<bool>();
    if (is_livox) {
      lidar_types.push_back(LIVOX);
      use_livox = true;
    } else {
      lidar_types.push_back(VLP);
      use_vlp = true;
    }
    lidar_topics_.push_back(lidar_i["topic"].as<std::string>());
    EP_LktoI_.emplace_back();
    EP_LktoI_.back().Init(lidar_i["Extrinsics"]);

    // only for ntu dataset
    // we consider lidar as the time baseline
    t_offset_lidar_ = EP_LktoI_.back().t_offset_ns * NS_TO_S;
    EP_LktoI_.back().t_offset_ns = 0;
  }
  

  for (int k = 0; k < num_lidars_; ++k) {
    lidar_max_timestamps_.push_back(0);
    Eigen::Matrix4d T_Lk_to_L0 = Eigen::Matrix4d::Identity();
    if (k > 0) {
      T_Lk_to_L0.block<3, 3>(0, 0) =
          (EP_LktoI_[0].q.inverse() * EP_LktoI_[k].q).toRotationMatrix();
      T_Lk_to_L0.block<3, 1>(0, 3) =
          EP_LktoI_[0].q.inverse() * (EP_LktoI_[k].p - EP_LktoI_[0].p);

      std::cout << "lidar " << k << "\n" << T_Lk_to_L0 << std::endl;
    }
    T_LktoL0_vec_.push_back(T_Lk_to_L0);
  }

  if (use_livox)
    livox_feature_extraction_ =
        std::make_shared<LivoxFeatureExtraction>(lidar_node, mode);
  if (use_vlp)
    velodyne_feature_extraction_ =
        std::make_shared<VelodyneFeatureExtraction>(lidar_node, mode);

  if (MODE::Odometry_Offline == mode) {
    std::cout << YELLOW << "\t- offline mode\n" << RESET << std::endl;
    LoadBag(node);
  } else {
    // IMU
    std::cout << YELLOW << "\t- online mode\n" << RESET << std::endl;
    sub_imu_ = nh.subscribe<sensor_msgs::Imu>(
        imu_topic_, 0, &MsgManager::IMUMsgHandle, this,
        ros::TransportHints().tcpNoDelay());
    // LiDAR
    for (size_t i = 0; i < lidar_topics_.size(); ++i) {
      if (lidar_types[i] == VLP) {
        subs_vlp16_.push_back(nh.subscribe<sensor_msgs::PointCloud2>(
            lidar_topics_[i], 0,
            boost::bind(&MsgManager::VelodyneMsgHandle, this, _1, i)));
      } else {
        subs_livox_.push_back(nh.subscribe<livox_ros_driver::CustomMsg>(
            lidar_topics_[i], 0,
            boost::bind(&MsgManager::LivoxMsgHandle, this, _1, i)));
      }
    }
    // Camera: Image feature track assigned to FeatureTrackerNode

    std::cout << "Subscribe topics:\n";
    std::cout << "\t- imu_topic: " << imu_topic_ << "\n";
    for (auto v : lidar_topics_) {
      std::cout << "\t- lidar_topic: " << v << "\n";
    }
    std::cout << "\t- image_topic: " << image_topic_ << "\n";
  }
}

void MsgManager::LoadBag(const YAML::Node& node) {
  double bag_start = node["bag_start"].as<double>();
  double bag_durr = node["bag_durr"].as<double>();

  std::vector<std::string> topics;
  topics.push_back(imu_topic_);  // IMU
  if (use_image_)                // Image
    topics.push_back(image_topic_);
  for (auto& v : lidar_topics_)  // LiDAR
    topics.push_back(v);

  bag_.open(bag_path_, rosbag::bagmode::Read);

  rosbag::View view_full;
  view_full.addQuery(bag_);
  ros::Time time_start = view_full.getBeginTime();
  time_start += ros::Duration(bag_start);
  ros::Time time_finish = (bag_durr < 0) ? view_full.getEndTime()
                                         : time_start + ros::Duration(bag_durr);
  view_.addQuery(bag_, rosbag::TopicQuery(topics), time_start, time_finish);
  if (view_.size() == 0) {
    ROS_ERROR("No messages to play on specified topics.  Exiting.");
    ros::shutdown();
    return;
  }

  std::cout << "LoadBag " << bag_path_ << " start at " << bag_start
            << " with duration " << (time_finish - time_start).toSec() << "\n";
  LOG(INFO) << "LoadBag " << bag_path_ << " start at " << bag_start
            << " with duration " << (time_finish - time_start).toSec();
}

void MsgManager::SpinBagOnce() {
  static rosbag::View::iterator view_iterator = view_.begin();
  if (view_iterator == view_.end()) {
    has_valid_msg_ = false;
    LOG(INFO) << "End of bag";
    return;
  }

  t_image_ms_ = 0;
  t_lidar_ms_ = 0;

  const rosbag::MessageInstance& m = *view_iterator;
  std::string msg_topic = m.getTopic();

  /// IMU
  if (msg_topic == imu_topic_) {
    sensor_msgs::Imu::ConstPtr imu_msg = m.instantiate<sensor_msgs::Imu>();
    IMUMsgHandle(imu_msg);
  }
  /// Cameara
  else if (msg_topic == image_topic_) {
    TicToc timer;
    if (m.isType<sensor_msgs::Image>()) {
      auto img_msg = m.instantiate<sensor_msgs::Image>();
      if (use_image_) ImageMsgHandle(img_msg);
    }
    t_image_ms_ = timer.toc();
  }
  /// LiDAR
  else if (std::find(lidar_topics_.begin(), lidar_topics_.end(), msg_topic) !=
           lidar_topics_.end()) {
    TicToc timer;
    auto it = std::find(lidar_topics_.begin(), lidar_topics_.end(), msg_topic);
    auto idx = std::distance(lidar_topics_.begin(), it);
    /// VLP
    if (lidar_types[idx] == VLP) {
      if (!m.isType<sensor_msgs::PointCloud2>()) std::cout << "Wrong type\n";

      auto lidar_msg = m.instantiate<sensor_msgs::PointCloud2>();
      VelodyneMsgHandle(lidar_msg, idx);
    }
    /// LIVOX
    else if (lidar_types[idx] == LIVOX) {
      if (!m.isType<livox_ros_driver::CustomMsg>()) std::cout << "Wrong type\n";

      auto lidar_msg = m.instantiate<livox_ros_driver::CustomMsg>();
      LivoxMsgHandle(lidar_msg, idx);
    }
    t_lidar_ms_ = timer.toc();
  }

  view_iterator++;
}

void MsgManager::LogInfo() const {
  int m_size[3] = {0, 0, 0};
  m_size[0] = imu_buf_.size();
  m_size[1] = lidar_buf_.size();
  if (use_image_) m_size[2] = feature_tracker_node_->NumImageMsg();
  LOG(INFO) << "imu/lidar/image msg left: " << m_size[0] << "/" << m_size[1]
            << "/" << m_size[2];
}

void MsgManager::RemoveBeginData(double start_time,
                                 double relative_start_time) {
  for (auto iter = lidar_buf_.begin(); iter != lidar_buf_.end();) {
    if (iter->is_time_wrt_traj_start && iter->timestamp < relative_start_time) {
      if (iter->max_timestamp <= relative_start_time) {
        // 1 remove the whole scan
        iter = lidar_buf_.erase(iter);
        continue;
      } else {
        // 2 remove part of the scan
        double t_aft = relative_start_time + 1e-3;
        LiDARCloudData scan_bef, scan_aft;
        scan_aft.timestamp = t_aft;
        scan_aft.max_timestamp = iter->max_timestamp;
        pcl::FilterCloudByTimestamp(iter->raw_cloud, t_aft, scan_bef.raw_cloud,
                                    scan_aft.raw_cloud);
        pcl::FilterCloudByTimestamp(iter->surf_cloud, t_aft,
                                    scan_bef.surf_cloud, scan_aft.surf_cloud);
        pcl::FilterCloudByTimestamp(iter->corner_cloud, t_aft,
                                    scan_bef.corner_cloud,
                                    scan_aft.corner_cloud);

        iter->timestamp = t_aft;
        *iter->raw_cloud = *scan_aft.raw_cloud;
        *iter->surf_cloud = *scan_aft.surf_cloud;
        *iter->corner_cloud = *scan_aft.corner_cloud;
      }
    }

    iter++;
  }

  if (use_image_) {
    auto& img_feature_buf = feature_tracker_node_->GetImageFeatureBuf();
    while (!img_feature_buf.empty() &&
           img_feature_buf.front()->header.stamp.toSec() + t_offset_camera_ <
               start_time) {
      img_feature_buf.pop_front();
    }
  }
}

bool MsgManager::HasEnvMsg() const {
  int env_msg = lidar_buf_.size();
  if (cur_imu_timestamp_ < 0 && env_msg > 100)
    LOG(WARNING) << "No IMU data. CHECK imu topic" << imu_topic_;

  return env_msg > 0;
}

bool MsgManager::CheckMsgIsReady(double traj_max, double start_time,
                                 double knot_dt, bool in_scan_unit) const {
  double t_imu_wrt_start = cur_imu_timestamp_ - start_time;
  t_imu_wrt_start += t_offset_imu_;

  // if we do not have enough IMU mea.
  if (t_imu_wrt_start < traj_max) {
    return false;
  }

  double t_front_lidar = -1;
  // Count how many unique lidar streams
  std::vector<int> unique_lidar_ids;
  for (const auto& data : lidar_buf_) {
    if (std::find(unique_lidar_ids.begin(), unique_lidar_ids.end(),
                  data.lidar_id) != unique_lidar_ids.end())
      continue;
    unique_lidar_ids.push_back(data.lidar_id);

    t_front_lidar = std::max(t_front_lidar, data.max_timestamp);
  }

  // if we have mea. of every unique lidar
  if ((int)unique_lidar_ids.size() != num_lidars_) return false;

  double t_back_lidar = lidar_max_timestamps_[0];
  for (auto t : lidar_max_timestamps_) {
    t_back_lidar = std::min(t_back_lidar, t);
  }

  // if we have enough scans
  if (in_scan_unit) {
    if (t_front_lidar > t_imu_wrt_start) return false;
  } else {
    if (t_back_lidar < traj_max) return false;
  }

  if (use_image_) {
    const auto& img_buf = feature_tracker_node_->GetImageFeatureBuf();
    if (img_buf.empty()) return false;

    double t_back_image = img_buf.back()->header.stamp.toSec() - start_time;
    t_back_image += t_offset_camera_;

    if (in_scan_unit) {
      double t_next_seg = std::ceil(t_front_lidar / knot_dt) * knot_dt;
      if (t_back_image < t_next_seg) return false;
    } else {
      if (t_back_image < traj_max) return false;
    }
  }

  return true;
}

bool MsgManager::AddToNextMsg(std::deque<LiDARCloudData>::iterator scan,
                              double traj_max, bool in_scan_unit) {
  bool add_entire_scan = false;
  if (scan->timestamp > traj_max) return add_entire_scan;

  if (in_scan_unit || scan->max_timestamp < traj_max) {
    *next_msgs.lidar_raw_cloud += (*scan->raw_cloud);
    *next_msgs.lidar_surf_cloud += (*scan->surf_cloud);
    *next_msgs.lidar_corner_cloud += (*scan->corner_cloud);

    if (next_msgs.scan_num == 0) {
      // first scan
      next_msgs.lidar_timestamp = scan->timestamp;
      next_msgs.lidar_max_timestamp = scan->max_timestamp;
    } else {
      next_msgs.lidar_timestamp =
          std::min(next_msgs.lidar_timestamp, scan->timestamp);
      next_msgs.lidar_max_timestamp =
          std::max(next_msgs.lidar_max_timestamp, scan->max_timestamp);
    }

    add_entire_scan = true;
  } else {
    // get part of a scan
    LiDARCloudData scan_bef, scan_aft;
    pcl::FilterCloudByTimestamp(scan->raw_cloud, traj_max, scan_bef.raw_cloud,
                                scan_aft.raw_cloud);
    pcl::FilterCloudByTimestamp(scan->surf_cloud, traj_max, scan_bef.surf_cloud,
                                scan_aft.surf_cloud);
    pcl::FilterCloudByTimestamp(scan->corner_cloud, traj_max,
                                scan_bef.corner_cloud, scan_aft.corner_cloud);
    // scan_bef
    scan_bef.timestamp = scan->timestamp;
    scan_bef.max_timestamp = traj_max - 1e-9;
    scan_aft.timestamp = traj_max;
    scan_aft.max_timestamp = scan->max_timestamp;

    // save the part of the for next update
    scan->timestamp = traj_max;
    // *scan.max_timestamp = ； // the max time is right already
    *scan->raw_cloud = *scan_aft.raw_cloud;
    *scan->surf_cloud = *scan_aft.surf_cloud;
    *scan->corner_cloud = *scan_aft.corner_cloud;

    *next_msgs.lidar_raw_cloud += (*scan_bef.raw_cloud);
    *next_msgs.lidar_surf_cloud += (*scan_bef.surf_cloud);
    *next_msgs.lidar_corner_cloud += (*scan_bef.corner_cloud);

    // update the timestamp of next_msgs
    if (next_msgs.scan_num == 0) {
      // first scan
      next_msgs.lidar_timestamp = scan_bef.timestamp;
      next_msgs.lidar_max_timestamp = scan_bef.max_timestamp;
    } else {
      next_msgs.lidar_timestamp =
          std::min(next_msgs.lidar_timestamp, scan_bef.timestamp);
      next_msgs.lidar_max_timestamp =
          std::max(next_msgs.lidar_max_timestamp, scan_bef.max_timestamp);
    }

    add_entire_scan = false;
  }

  // caculate the num of involved scans in next_msgs
  next_msgs.scan_num++;

  return add_entire_scan;
}

bool MsgManager::GetNextMsgs(double traj_max, double start_time, double knot_dt,
                             bool in_scan_unit) {
  assert(start_time > 0);
  next_msgs.Clear();

  // if we only have IMU mea.
  if (!HasEnvMsg()) return false;

  for (auto& data : lidar_buf_) {
    if (!data.is_time_wrt_traj_start) {
      data.ToRelativeMeasureTime(start_time);
      lidar_max_timestamps_[data.lidar_id] = data.max_timestamp;

      // std::cout << data.timestamp << ", " << data.max_timestamp << "\n";
    }
  }
  // std::sort(lidar_buf_.begin(), lidar_buf_.end());

  double t_relative_start = 0;
  RemoveBeginData(start_time, t_relative_start);

  if (!CheckMsgIsReady(traj_max, start_time, knot_dt, in_scan_unit))
    return false;

  std::vector<int> unique_lidar_ids;
  for (auto it = lidar_buf_.begin(); it != lidar_buf_.end();) {
    if (it->timestamp > traj_max) {
      ++it;
      continue;
    }

    if (std::find(unique_lidar_ids.begin(), unique_lidar_ids.end(),
                  it->lidar_id) == unique_lidar_ids.end()) {
      unique_lidar_ids.push_back(it->lidar_id);
    }
    bool add_entire_scan = AddToNextMsg(it, traj_max, in_scan_unit);

    // std::cout << "[Add new msg] lidar id: " << it->lidar_id << ";  t in ["
    //           << it->timestamp << ", " << it->max_timestamp
    //           << "]; size: " << it->raw_cloud->size()
    //           << "; next_msgs.lidar_raw_cloud  "
    //           << next_msgs.lidar_raw_cloud->size() << "\n";

    if (add_entire_scan) {
      it = lidar_buf_.erase(it);
    } else {
      ++it;
    }

    if (in_scan_unit && (int)unique_lidar_ids.size() == num_lidars_) break;
  }

  if (use_image_) {
    double t_next_seg =
        std::ceil(next_msgs.lidar_max_timestamp / knot_dt) * knot_dt;
    double t_msg_max = in_scan_unit ? t_next_seg : traj_max;
    t_msg_max += start_time;

    auto& img_feature_buf = feature_tracker_node_->GetImageFeatureBuf();
    while (!img_feature_buf.empty() &&
           img_feature_buf.front()->header.stamp.toSec() + t_offset_camera_ <
               t_msg_max) {
      next_msgs.image_feature_msgs.emplace_back(img_feature_buf.front());
      img_feature_buf.pop_front();
    }
  }

  return true;
}

void MsgManager::IMUMsgHandle(const sensor_msgs::Imu::ConstPtr& imu_msg) {
  double t_last = cur_imu_timestamp_;
  cur_imu_timestamp_ = imu_msg->header.stamp.toSec() - add_extra_timeoffset_s_;

  // remove bad imu measurements
  if (t_last > 0 && cur_imu_timestamp_ - t_last > 4 * imu_period_s_) {
    // std::cout << RED << "[IMUMsgHandle] remove imu at " << cur_imu_timestamp_
    //           << " with dt " << cur_imu_timestamp_ - t_last << "\n"
    //           << RESET;
    return;
  }

  IMUData data;
  IMUMsgToIMUData(imu_msg, data);

  data.timestamp -= add_extra_timeoffset_s_;

  // for trajectory_manager
  imu_buf_.emplace_back(data);
}

void MsgManager::VelodyneMsgHandle(
    const sensor_msgs::PointCloud2::ConstPtr& vlp16_msg, int lidar_id) {
  RTPointCloud::Ptr vlp_raw_cloud(new RTPointCloud);
  velodyne_feature_extraction_->ParsePointCloud(vlp16_msg, vlp_raw_cloud);

  // transform the input cloud to Lidar0 frame
  if (lidar_id != 0)
    pcl::transformPointCloud(*vlp_raw_cloud, *vlp_raw_cloud,
                             T_LktoL0_vec_[lidar_id]);

  velodyne_feature_extraction_->LidarHandler(vlp_raw_cloud);

  lidar_buf_.emplace_back();
  lidar_buf_.back().lidar_id = lidar_id;
  // lidar_buf_.back().timestamp = vlp16_msg->header.stamp.toSec();
  lidar_buf_.back().timestamp = vlp16_msg->header.stamp.toSec() + t_offset_lidar_;
  lidar_buf_.back().raw_cloud = vlp_raw_cloud;
  lidar_buf_.back().surf_cloud =
      velodyne_feature_extraction_->GetSurfaceFeature();
  lidar_buf_.back().corner_cloud =
      velodyne_feature_extraction_->GetCornerFeature();
}

void MsgManager::LivoxMsgHandle(
    const livox_ros_driver::CustomMsg::ConstPtr& livox_msg, int lidar_id) {
  RTPointCloud::Ptr livox_raw_cloud(new RTPointCloud);
  livox_feature_extraction_->ParsePointCloud(livox_msg, livox_raw_cloud);

  LiDARCloudData data;
  data.lidar_id = lidar_id;
  data.timestamp = livox_msg->header.stamp.toSec();
  data.raw_cloud = livox_raw_cloud;
  data.surf_cloud = livox_feature_extraction_->GetSurfaceFeature();
  data.corner_cloud = livox_feature_extraction_->GetCornerFeature();
  lidar_buf_.push_back(data);

  if (lidar_id != 0) {
    pcl::transformPointCloud(*data.raw_cloud, *data.raw_cloud,
                             T_LktoL0_vec_[lidar_id]);
    pcl::transformPointCloud(*data.surf_cloud, *data.surf_cloud,
                             T_LktoL0_vec_[lidar_id]);
    pcl::transformPointCloud(*data.corner_cloud, *data.corner_cloud,
                             T_LktoL0_vec_[lidar_id]);
  }
}

}  // namespace clic
