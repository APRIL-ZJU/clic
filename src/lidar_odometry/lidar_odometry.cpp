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

#include <pcl/kdtree/impl/kdtree_flann.hpp>  // pcl::KdTreeFLANN

#include <lidar_odometry/lidar_odometry.h>
#include <lidar_odometry/livox_feature_extraction.h>
#include <lidar_odometry/velodyne_feature_extraction.h>
#include <pcl/common/transforms.h>            //pcl::transformPointCloud
#include <pcl_conversions/pcl_conversions.h>  // pcl::fromROSMsg

#include <algorithm>  // std::find

namespace clic {

struct by_timestamp {
  bool operator()(const PointCorrespondence& left,
                  const PointCorrespondence& right) {
    return left.t_point < right.t_point;
  }
};

LidarOdometry::LidarOdometry(const YAML::Node& node, Trajectory::Ptr traj)
    : trajectory_(std::move(traj)),
      use_corner_feature_(true),
      cor_downsample_(1),
      update_full_cloud_(false),
      kdtree_corner_map_(new pcl::KdTreeFLANN<PosPoint>()),
      kdtree_surface_map_(new pcl::KdTreeFLANN<PosPoint>()),
      cloud_key_pos_(new PosCloud),
      cloud_key_pos_xy_(new PosCloud),
      key_frame_updated_(false) {
  use_corner_feature_ = yaml::GetBool(node, "use_corner_feature", false);

  const YAML::Node& map_node = node["map_param"];
  keyframe_search_radius_ = map_node["keyframe_search_radius"].as<float>();
  keyframe_search_time_ = map_node["keyframe_search_time"].as<float>();
  keyframe_density_map_ = map_node["keyframe_density"].as<float>();

  cloud_reserved_time_ =
      yaml::GetValue<double>(map_node, "cloud_reserved_time", -1);
  std::cout << YELLOW
            << "\t- cloud reserved time in odom: " << cloud_reserved_time_
            << RESET << "\n";

  const YAML::Node& cur_scan_node = node["current_scan_param"];
  edge_min_valid_num_ = cur_scan_node["edge_min_valid_num"].as<int>();
  surf_min_valid_num_ = cur_scan_node["surf_min_valid_num"].as<int>();
  corner_leaf_size_ = cur_scan_node["corner_leaf_size"].as<float>();
  surface_leaf_size_ = cur_scan_node["surface_leaf_size"].as<float>();
  cor_downsample_ = cur_scan_node["correspondence_downsample"].as<int>();

  const YAML::Node& keyframe_node = node["keyframe_strategy"];
  keyframe_angle_degree_ = keyframe_node["angle_degree"].as<double>();
  keyframe_dist_meter_ = keyframe_node["dist_meter"].as<double>();
  keyframe_time_second_ = keyframe_node["time_second"].as<double>();
}

// Note: the timestamp of each point is the exact measured time
void LidarOdometry::FeatureCloudHandler(
    const clic::feature_cloud::ConstPtr& feature_msg) {
  RTPointCloud::Ptr corner_cloud(new RTPointCloud);
  RTPointCloud::Ptr surface_cloud(new RTPointCloud);
  RTPointCloud::Ptr full_cloud(new RTPointCloud);

  pcl::fromROSMsg(feature_msg->corner_cloud, *corner_cloud);
  pcl::fromROSMsg(feature_msg->surface_cloud, *surface_cloud);
  pcl::fromROSMsg(feature_msg->full_cloud, *full_cloud);

  feature_cur_.Clear();

  double scan_time = feature_msg->header.stamp.toSec();
  double traj_start_time = trajectory_->GetDataStartTime();
  feature_cur_.timestamp = scan_time - traj_start_time;
  CloudToRelativeMeasureTime(corner_cloud, scan_time, traj_start_time);
  CloudToRelativeMeasureTime(surface_cloud, scan_time, traj_start_time);
  CloudToRelativeMeasureTime(full_cloud, scan_time, traj_start_time);

  if (use_corner_feature_) {
    pcl::RTPointCloudToPosCloud(corner_cloud, feature_cur_.corner_features);
  }
  pcl::RTPointCloudToPosCloud(surface_cloud, feature_cur_.surface_features);
  pcl::RTPointCloudToPosCloud(full_cloud, feature_cur_.full_cloud,
                              &feature_cur_.time_max);
  /// Downsample current scan
  DownsampleLiDARFeature(feature_cur_, feature_cur_ds_);
}

// Note: the timestamp of each point is the exact measured time
void LidarOdometry::FeatureCloudHandler(const LiDARFeature& lidar_feature) {
  feature_cur_.Clear();
  feature_cur_.time_max = lidar_feature.time_max;
  feature_cur_.timestamp = lidar_feature.timestamp;
  *(feature_cur_.corner_features) = *(lidar_feature.corner_features);
  *(feature_cur_.surface_features) = *(lidar_feature.surface_features);
  *(feature_cur_.full_cloud) = *(lidar_feature.full_cloud);

  /// Downsample current scan
  DownsampleLiDARFeature(feature_cur_, feature_cur_ds_);
}

// Note: the timestamp of each point is the exact measured time
void LidarOdometry::FeatureCloudHandler(double scan_timestamp,
                                        double scan_time_max,
                                        const RTPointCloud::Ptr corner_feature,
                                        const RTPointCloud::Ptr surface_feature,
                                        const RTPointCloud::Ptr raw_cloud) {
  feature_cur_.Clear();
  feature_cur_.timestamp = scan_timestamp;
  feature_cur_.time_max = scan_time_max;
  if (use_corner_feature_) {
    pcl::RTPointCloudToPosCloud(corner_feature, feature_cur_.corner_features);
  }
  pcl::RTPointCloudToPosCloud(surface_feature, feature_cur_.surface_features);
  pcl::RTPointCloudToPosCloud(raw_cloud, feature_cur_.full_cloud);

  DownsampleLiDARFeature(feature_cur_, feature_cur_ds_);
}

bool LidarOdometry::UpdateMapData() {
  UpdateKeyFrames();
  if (key_frame_updated_) {
    key_frame_updated_ = false;
    ExtractSurroundFeatures(feature_cur_.timestamp);
    return true;
    // std::cout << "Pause " << std::endl << std::getchar();
  } else {
    return false;
  }
}

void LidarOdometry::GetLoamFeatureAssociation() {
  LiDARFeature cur_lf_in_map;

  double target_time = feature_map_ds_.timestamp;

  bool in_target_frame = false;
  if (in_target_frame) {
    if (use_corner_feature_) {
      trajectory_->UndistortScan(*feature_cur_ds_.corner_features, target_time,
                                 *cur_lf_in_map.corner_features);
    }
    trajectory_->UndistortScan(*feature_cur_ds_.surface_features, target_time,
                               *cur_lf_in_map.surface_features);
  } else {
    if (use_corner_feature_) {
      trajectory_->UndistortScanInG(*feature_cur_ds_.corner_features,
                                    *cur_lf_in_map.corner_features);
    }
    trajectory_->UndistortScanInG(*feature_cur_ds_.surface_features,
                                  *cur_lf_in_map.surface_features);
  }

  FindCorrespondence(feature_cur_ds_, cur_lf_in_map);

  std::sort(point_correspondence_.begin(), point_correspondence_.end(),
            by_timestamp());

  DownSampleCorrespondence();

  LOG(INFO) << "point correspondence : " << point_correspondence_.size()
            << "; point correspondence_ds : "
            << point_correspondence_ds_.size();
}

bool LidarOdometry::UpdateKeyFrames() {
  // cache current scan
  LiDARFeature feature;
  feature = feature_cur_;
  cache_feature_container_[feature.timestamp] = feature;

  // update keyscan
  for (auto iter = cache_feature_container_.begin();
       iter != cache_feature_container_.end();) {
    /// remove non-keyscan
    if (!IsKeyFrame(iter->second.time_max)) {
      // the traj is still in optimization
      if (iter->second.time_max >
          trajectory_->GetActiveTime()) {
        break;
      } else {
        cache_feature_container_.erase(iter++);
        continue;
      }
    }


    LiDARFeature key_scan;
    key_scan.timestamp = iter->second.timestamp;
    key_scan.time_max = iter->second.time_max;
    if (local_feature_container_.empty()) {
      // we assume the fist scan is static
      // key_pose.timestamp = trajectory_->minTime(LiDARSensor);
      *(key_scan.corner_features) = *(iter->second.corner_features);
      *(key_scan.surface_features) = *(iter->second.surface_features);
      *(key_scan.full_cloud) = *(iter->second.full_cloud);
    } else {
      // the timestamp of each point is wrt. the start time of the scan
      trajectory_->UndistortScan(*(iter->second.corner_features),
                                 key_scan.timestamp,
                                 *(key_scan.corner_features));
      trajectory_->UndistortScan(*(iter->second.surface_features),
                                 key_scan.timestamp,
                                 *(key_scan.surface_features));
      trajectory_->UndistortScan(*(iter->second.full_cloud), key_scan.timestamp,
                                 *(key_scan.full_cloud));
    }
#if false
    if (local_feature_container_.size() < 10)
      local_feature_container_[key_scan.timestamp] = key_scan;
    else {
      LiDARFeature key_scan_ds;
      DownsampleLiDARFeature(key_scan, key_scan_ds);
      local_feature_container_[key_scan.timestamp] = key_scan_ds;
    }
#else
    LiDARFeature key_scan_ds;
    DownsampleLiDARFeature(key_scan, key_scan_ds);
    local_feature_container_[key_scan.timestamp] = key_scan;
    local_feature_container_all_ds_[key_scan.timestamp] = key_scan_ds;
#endif
    cache_feature_container_.erase(iter++);

    PosPoint key_pose;
    key_pose.timestamp = key_scan.timestamp;
    Eigen::Vector3d p =
        trajectory_->GetLidarPose(key_scan.timestamp).translation();
    key_pose.x = p[0];
    key_pose.y = p[1];
    key_pose.z = p[2];
    cloud_key_pos_->push_back(key_pose);

    key_pose.z = 0;
    cloud_key_pos_xy_->push_back(key_pose);

    trajectory_->SetForcedFixedTime(key_scan.time_max);

    key_frame_updated_ = true;
  }

  // remove old keyscans (100s)
  if (cloud_reserved_time_ > 0) {
    double t_now = local_feature_container_.rbegin()->first;
    auto iter = local_feature_container_.begin();
    while (iter != local_feature_container_.end()) {
      if (t_now - (*iter).first > cloud_reserved_time_) {
        iter = local_feature_container_.erase(iter);
      } else {
        break;
      }
    }
  }

  return key_frame_updated_;
}

const bool LidarOdometry::IsKeyFrame(double time) const {
  // first scan
  if (cloud_key_pos_->points.empty()) return true;

  // check if the trajectory is static
  if (time > trajectory_->GetActiveTime()) return false;

  if ((time - cloud_key_pos_->back().timestamp) > keyframe_time_second_)
    return true;

  SE3d pose_cur = trajectory_->GetLidarPose(time);
  SE3d pose_last = trajectory_->GetLidarPose(cloud_key_pos_->back().timestamp);

  SE3d pose_cur_to_last = pose_last.inverse() * pose_cur;
  Eigen::AngleAxisd v(pose_cur_to_last.so3().unit_quaternion());

  double dist_meter = pose_cur_to_last.translation().norm();
  double angle_degree = v.angle() * (180. / M_PI);

  if (angle_degree > keyframe_angle_degree_ ||
      dist_meter > keyframe_dist_meter_) {
    //    LOG(INFO) << WHITE << "[ IsKeyFrame ] " << angle_degree << " : "
    //              << dist_meter;
    return true;
  }

  return false;
}

void LidarOdometry::GetNearTimeKeyScanID(PosCloud::Ptr key_pos_selected,
                                         const double cur_time) const {
  std::vector<double> t_added;
  t_added.reserve(key_pos_selected->size());
  for (size_t i = 0; i < key_pos_selected->size(); i++) {
    t_added.push_back(key_pos_selected->points[i].timestamp);
  }

  /// temporally nearby 
  float map_start_time = cur_time - keyframe_search_time_;
  for (int i = cloud_key_pos_->size() - 1; i >= 0; i--) {
    if (cloud_key_pos_->points[i].timestamp < map_start_time) break;

    auto iter = std::find(t_added.begin(), t_added.end(),
                          cloud_key_pos_->points[i].timestamp);
    if (t_added.end() == iter) {
      key_pos_selected->push_back(cloud_key_pos_->points[i]);
    }
  }
}

void LidarOdometry::GetNearDistKeyScanID(PosCloud::Ptr key_pos_selected) {
  PosCloud::Ptr nearby_key_pos(new PosCloud);

  std::vector<int> indices;
  std::vector<float> sqr_dists;
  pcl::KdTreeFLANN<PosPoint>::Ptr kdtree_nearby_key_pos(
      new pcl::KdTreeFLANN<PosPoint>());
  kdtree_nearby_key_pos->setInputCloud(cloud_key_pos_);
  kdtree_nearby_key_pos->radiusSearch(cloud_key_pos_->back(),
                                      (double)keyframe_search_radius_, indices,
                                      sqr_dists);
  /// spatially nearby
  for (const int& idx : indices) {
    nearby_key_pos->push_back(cloud_key_pos_->points[idx]);
  }

  if (cloud_key_pos_->size() > 20) {
    VoxelFilter<PosPoint> surrounding_key_pos_voxel_filter;
    surrounding_key_pos_voxel_filter.SetResolution(keyframe_density_map_);
    surrounding_key_pos_voxel_filter.SetInputCloud(nearby_key_pos);
    surrounding_key_pos_voxel_filter.Filter(key_pos_selected);
  } else {
    key_pos_selected = nearby_key_pos;
  }
}

void LidarOdometry::TransformLiDARFeature(const LiDARFeature& lf_in,
                                          const Eigen::Matrix4d& T_in_to_out,
                                          LiDARFeature& lf_out) const {
  if (use_corner_feature_) {
    pcl::transformPointCloud(*lf_in.corner_features, *lf_out.corner_features,
                             T_in_to_out);
  }
  pcl::transformPointCloud(*lf_in.surface_features, *lf_out.surface_features,
                           T_in_to_out);

  if (update_full_cloud_) {
    pcl::transformPointCloud(*lf_in.full_cloud, *lf_out.full_cloud,
                             T_in_to_out);
  }

  lf_out.timestamp = lf_in.timestamp;
  lf_out.time_max = lf_in.time_max;
}

void LidarOdometry::DownsampleLiDARFeature(const LiDARFeature& lf_in,
                                           LiDARFeature& lf_out) const {
  lf_out.Clear();
  lf_out.timestamp = lf_in.timestamp;
  lf_out.time_max = lf_in.time_max;

  if (use_corner_feature_) {
    VoxelFilter<PosPoint> corner_features_voxel_filter;
    corner_features_voxel_filter.SetResolution(corner_leaf_size_);
    corner_features_voxel_filter.SetInputCloud(lf_in.corner_features);
    corner_features_voxel_filter.Filter(lf_out.corner_features);
  }

  VoxelFilter<PosPoint> surface_features_voxel_filter;
  surface_features_voxel_filter.SetResolution(surface_leaf_size_);
  surface_features_voxel_filter.SetInputCloud(lf_in.surface_features);
  surface_features_voxel_filter.Filter(lf_out.surface_features);
}

void LidarOdometry::ExtractSurroundFeatures(const double cur_time) {
  assert(!cloud_key_pos_->points.empty() && "no key scan to build local map");

  PosCloud::Ptr key_pos_selected(new PosCloud);
  GetNearDistKeyScanID(key_pos_selected);
  GetNearTimeKeyScanID(key_pos_selected, cur_time);

  double target_time = cloud_key_pos_->back().timestamp;
  feature_map_.Clear();
  feature_map_.timestamp = target_time;

  bool in_target_frame = false;
  // first scan
  if (local_feature_container_.size() == 1) {
    double kf_time = target_time;
    // transform scan from local frame to global frame
    SE3d pose_cur_to_G = trajectory_->GetLidarPose(kf_time);  // TG_Lcur
    LiDARFeature lf_out;
    if (in_target_frame) {
      TransformLiDARFeature(local_feature_container_[kf_time],
                            Eigen::Matrix4d::Identity(), lf_out);
    } else {
      TransformLiDARFeature(local_feature_container_[kf_time],
                            pose_cur_to_G.matrix(), lf_out);
    }

    if (use_corner_feature_) {
      *feature_map_.corner_features += *(lf_out.corner_features);
    }
    *feature_map_.surface_features += *(lf_out.surface_features);
    if (update_full_cloud_) {
      *feature_map_.full_cloud += *(lf_out.full_cloud);
    }
  } else {
    SE3d pose_G_to_target =
        trajectory_->GetLidarPose(target_time).inverse();  // TLtar_G
    for (auto const& pos : key_pos_selected->points) {
      double kf_time = pos.timestamp;
      if (local_feature_container_.find(kf_time) !=
          local_feature_container_.end()) {
      
        SE3d pose_cur_to_G = trajectory_->GetLidarPose(kf_time);  // TG_Lcur
        Eigen::Matrix4d cur_to_target =
            (pose_G_to_target * pose_cur_to_G).matrix();  // TLtar_Lcur

        LiDARFeature lf_out;
        if (in_target_frame) {
          TransformLiDARFeature(local_feature_container_[kf_time],
                                cur_to_target, lf_out);
        } else {
          TransformLiDARFeature(local_feature_container_[kf_time],
                                pose_cur_to_G.matrix(), lf_out);
        }

        if (use_corner_feature_) {
          *feature_map_.corner_features += *lf_out.corner_features;
        }
        *feature_map_.surface_features += *lf_out.surface_features;
        if (update_full_cloud_) {
          *feature_map_.full_cloud += *lf_out.full_cloud;
        }

        feature_map_.time_max =
            std::max(feature_map_.time_max, lf_out.time_max);
      }
    }
  }

  if (feature_map_.surface_features->size() < 10000) {
    feature_map_ds_.Clear();
    feature_map_ds_ = feature_map_;
  } else {
    DownsampleLiDARFeature(feature_map_, feature_map_ds_);
  }

  LOG(INFO) << "[feature_map_ds] "
            << "surface cloud map : "
            << feature_map_ds_.surface_features->size()
            << "; cornor cloud map: " << feature_map_ds_.corner_features->size()
            << "; feature map time : [" << feature_map_ds_.timestamp << ", "
            << feature_map_ds_.time_max << "].";

  SetTargetMap(feature_map_ds_);
}

void LidarOdometry::SetTargetMap(const LiDARFeature& feature_map) {
  assert(!feature_map.surface_features->empty() &&
         "[SetTargetMap] input surface_features are empty");
  assert(!(use_corner_feature_ && feature_map.corner_features->empty()) &&
         "[SetTargetMap] input corner_features are empty");

  if (use_corner_feature_)
    kdtree_corner_map_->setInputCloud(feature_map.corner_features);
  kdtree_surface_map_->setInputCloud(feature_map.surface_features);
}

bool LidarOdometry::FindCorrespondence(const LiDARFeature& lf_cur,
                                       const LiDARFeature& lf_cur_in_M) {
  point_correspondence_.clear();
  point_correspondence_.reserve(lf_cur.surface_features->size());

  if ((use_corner_feature_ &&
       lf_cur.corner_features->size() < size_t(edge_min_valid_num_)) &&
      lf_cur.surface_features->size() < size_t(surf_min_valid_num_)) {
    LOG(WARNING) << "[FindCorrespondence] No enough feature points ! Corner: "
                 << lf_cur.corner_features->size()
                 << "; Surface: " << lf_cur.surface_features->size();
    return false;
  }

  /// corner feature 
  size_t corner_step = lf_cur.corner_features->size() / 1500 + 1;
  if (use_corner_feature_ && feature_map_ds_.corner_features->size() > 10) {
    for (size_t i = 0; i < lf_cur.corner_features->size(); i += corner_step) {
      PosPoint point_inM = lf_cur_in_M.corner_features->points[i];
      std::vector<int> k_indices;
      std::vector<float> k_sqr_dists;
      kdtree_corner_map_->nearestKSearch(point_inM, 5, k_indices, k_sqr_dists);

      if (k_sqr_dists[4] > 1.0) continue;

      cv::Mat matA1(3, 3, CV_32F, cv::Scalar::all(0));
      cv::Mat matD1(1, 3, CV_32F, cv::Scalar::all(0));
      cv::Mat matV1(3, 3, CV_32F, cv::Scalar::all(0));

      Eigen::Vector3d cp(0, 0, 0);
      for (int j = 0; j < 5; j++) {
        cp[0] += feature_map_ds_.corner_features->points[k_indices[j]].x;
        cp[1] += feature_map_ds_.corner_features->points[k_indices[j]].y;
        cp[2] += feature_map_ds_.corner_features->points[k_indices[j]].z;
      }
      cp /= 5.0;

      float a11 = 0, a12 = 0, a13 = 0, a22 = 0, a23 = 0, a33 = 0;
      for (int j = 0; j < 5; j++) {
        float ax =
            feature_map_ds_.corner_features->points[k_indices[j]].x - cp[0];
        float ay =
            feature_map_ds_.corner_features->points[k_indices[j]].y - cp[1];
        float az =
            feature_map_ds_.corner_features->points[k_indices[j]].z - cp[2];

        a11 += ax * ax;
        a12 += ax * ay;
        a13 += ax * az;
        a22 += ay * ay;
        a23 += ay * az;
        a33 += az * az;
      }
      a11 /= 5;
      a12 /= 5;
      a13 /= 5;
      a22 /= 5;
      a23 /= 5;
      a33 /= 5;

      matA1.at<float>(0, 0) = a11;
      matA1.at<float>(0, 1) = a12;
      matA1.at<float>(0, 2) = a13;
      matA1.at<float>(1, 0) = a12;
      matA1.at<float>(1, 1) = a22;
      matA1.at<float>(1, 2) = a23;
      matA1.at<float>(2, 0) = a13;
      matA1.at<float>(2, 1) = a23;
      matA1.at<float>(2, 2) = a33;

      cv::eigen(matA1, matD1, matV1);

      if (matD1.at<float>(0, 0) > 3 * matD1.at<float>(0, 1)) {
        Eigen::Vector3d normal(matV1.at<float>(0, 0), matV1.at<float>(0, 1),
                               matV1.at<float>(0, 2));
        // double plane_d = -normal.dot(cp);

        PointCorrespondence point_cor;
        point_cor.geo_type = Line;
        point_cor.t_point = lf_cur.corner_features->points[i].timestamp;
        point_cor.t_map = feature_map_.timestamp;
        point_cor.geo_point = cp;
        point_cor.geo_normal = normal;
        point_cor.point = Eigen::Vector3d(lf_cur.corner_features->points[i].x,
                                          lf_cur.corner_features->points[i].y,
                                          lf_cur.corner_features->points[i].z);
        point_correspondence_.push_back(point_cor);
      }
    }
  }
  int corner_feature_num = point_correspondence_.size();

  /// surl fearure
  map_corrs_viewer.clear();
  size_t surf_step = lf_cur.surface_features->size() / 1500 + 1;
  for (size_t i = 0; i < lf_cur.surface_features->size(); i += surf_step) {
    if (lf_cur.surface_features->points[i].timestamp < 0) continue;

    PosPoint point_inM = lf_cur_in_M.surface_features->points[i];
    std::vector<int> k_indices;
    std::vector<float> k_sqr_dists;
    kdtree_surface_map_->nearestKSearch(point_inM, 5, k_indices, k_sqr_dists);

    if (k_sqr_dists[4] > 1.0) continue;

    Eigen::Matrix<float, 5, 3> matA0;
    Eigen::Matrix<float, 5, 1> matB0;
    Eigen::Vector3f matX0;
    matA0.setZero();
    matB0.fill(-1);
    matX0.setZero();

    if (k_indices.size() < 5) continue;

    for (int j = 0; j < 5; j++) {
      matA0(j, 0) = feature_map_ds_.surface_features->points[k_indices[j]].x;
      matA0(j, 1) = feature_map_ds_.surface_features->points[k_indices[j]].y;
      matA0(j, 2) = feature_map_ds_.surface_features->points[k_indices[j]].z;
    }

    matX0 = matA0.colPivHouseholderQr().solve(matB0);

    float pa = matX0(0, 0);
    float pb = matX0(1, 0);
    float pc = matX0(2, 0);
    float pd = 1;

    float ps = sqrt(pa * pa + pb * pb + pc * pc);
    pa /= ps;
    pb /= ps;
    pc /= ps;
    pd /= ps;

    bool planeValid = true;
    for (int j = 0; j < 5; j++) {
      if (fabs(pa * feature_map_ds_.surface_features->points[k_indices[j]].x +
               pb * feature_map_ds_.surface_features->points[k_indices[j]].y +
               pc * feature_map_ds_.surface_features->points[k_indices[j]].z +
               pd) > 0.2) {
        planeValid = false;
        break;
      }
    }

    if (planeValid) {
      PointCorrespondence point_cor;
      point_cor.geo_type = Plane;
      point_cor.t_point = lf_cur.surface_features->points[i].timestamp;
      point_cor.t_map = feature_map_ds_.timestamp;
      point_cor.point = Eigen::Vector3d(lf_cur.surface_features->points[i].x,
                                        lf_cur.surface_features->points[i].y,
                                        lf_cur.surface_features->points[i].z);
      point_cor.geo_plane = Eigen::Vector4d(pa, pb, pc, pd);
      point_correspondence_.push_back(point_cor);

      double color =
          point_cor.point[0] + point_cor.point[1] + point_cor.point[2];

      for (int i = 0; i < 5; ++i) {
        VPoint vp;
        vp.x = matA0(i, 0);
        vp.y = matA0(i, 1);
        vp.z = matA0(i, 2);
        vp.intensity = color;
        map_corrs_viewer.push_back(vp);
      }
    }
  }

  int surf_feature_num = point_correspondence_.size() - corner_feature_num;
  LOG(INFO) << "point correspondence: " << point_correspondence_.size()
            << "; corner/surfel " << corner_feature_num << "/"
            << surf_feature_num;
  return true;
}

// TODO speed up
void LidarOdometry::DownSampleCorrespondence() {
  int step = cor_downsample_;
  // if (point_correspondence_.size() < 1200)
  // step = 1;

  size_t reserve_size = point_correspondence_.size() / step + 1;
  point_correspondence_ds_.clear();
  point_correspondence_ds_.reserve(reserve_size);
  for (size_t i = 0; i < point_correspondence_.size(); i += step) {
    point_correspondence_ds_.push_back(point_correspondence_.at(i));
  }
}

void LidarOdometry::FindNearbyKeyFrames(const int key, const int search_num,
                                        PosCloud::Ptr& cloud_in_G) const {
  cloud_in_G->clear();

  int keyscan_size = cloud_key_pos_xy_->size();
  for (int i = -search_num; i <= search_num; i++) {
    int key_index = key + i;
    if (key_index < 0 || key_index >= keyscan_size) continue;

    double kf_time = cloud_key_pos_xy_->points.at(key_index).timestamp;
    if (local_feature_container_all_ds_.find(kf_time) ==
        local_feature_container_all_ds_.end())
      continue;

    const auto& local_lf = local_feature_container_all_ds_.at(kf_time);

    PosCloud local_cloud, global_cloud;
    if (!local_lf.full_cloud->empty()) {
      local_cloud += *local_lf.full_cloud;
    } else {
      local_cloud += *local_lf.surface_features;
    }

    Eigen::Matrix4d T_cur_to_G = trajectory_->GetLidarPose(kf_time).matrix();
    pcl::transformPointCloud(local_cloud, global_cloud, T_cur_to_G);
    *cloud_in_G += global_cloud;
  }
}

void LidarOdometry::UpdateCloudKeyPos(
    const std::pair<int, int>& cur_wrt_history) {
  // update key_pose
  double t_cur = cloud_key_pos_xy_->points[cur_wrt_history.first].timestamp;
  double t_history =
      cloud_key_pos_xy_->points[cur_wrt_history.second].timestamp;

  for (size_t i = 0; i < cloud_key_pos_xy_->size(); i++) {
    double kf_time = cloud_key_pos_xy_->points[i].timestamp;
    if (kf_time < t_history || kf_time > t_cur) {
      continue;
    }

    Eigen::Vector3d p = trajectory_->GetLidarPose(kf_time).translation();
    cloud_key_pos_->points[i].x = p(0);
    cloud_key_pos_->points[i].y = p(1);
    cloud_key_pos_->points[i].z = p(2);
    cloud_key_pos_xy_->points[i].x = p(0);
    cloud_key_pos_xy_->points[i].y = p(1);
  }

  // inform odometry to update map
  key_frame_updated_ = true;
}

}  // namespace clic
