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

#include <clic/feature_cloud.h>
#include <lidar_odometry/lidar_feature.h>
#include <spline/trajectory.h>
#include <utils/cloud_tool.h>
#include <utils/yaml_utils.h>

namespace clic {

class LidarOdometry {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef std::shared_ptr<LidarOdometry> Ptr;

  LidarOdometry(const YAML::Node& node, Trajectory::Ptr traj);

  // online
  void FeatureCloudHandler(const clic::feature_cloud::ConstPtr& feature_msg);

  // offline
  void FeatureCloudHandler(const LiDARFeature& lidar_feature);

  void FeatureCloudHandler(double scan_timestamp, double scan_time_max,
                           const RTPointCloud::Ptr corner_feature,
                           const RTPointCloud::Ptr surface_feature,
                           const RTPointCloud::Ptr raw_cloud);

  bool UpdateMapData();

  void SetUpdateMap(bool update_map) { key_frame_updated_ = update_map; }

  void GetLoamFeatureAssociation();

  const Eigen::aligned_vector<PointCorrespondence>& GetPointCorrespondence()
      const {
    return point_correspondence_ds_;
  }

  // loop closure
  PosCloud::Ptr GetCloudKeyPos2D() const { return cloud_key_pos_xy_; };

  const PosCloud::Ptr GetCloudKeyPos3D() const { return cloud_key_pos_; };

  void FindNearbyKeyFrames(const int key, const int search_num,
                           PosCloud::Ptr& cloud_in_G) const;

  void UpdateCloudKeyPos(const std::pair<int, int>& cur_wrt_history);

  ///===== for visualization
  const LiDARFeature& GetFeatureCurrent() const { return feature_cur_; }

  const LiDARFeature& GetFeatureCurrentDs() const { return feature_cur_ds_; }

  const LiDARFeature& GetFeatureMap() const { return feature_map_; }

  const LiDARFeature& GetFeatureMapDs() const { return feature_map_ds_; }

  VPointCloud map_corrs_viewer;

 private:
  bool UpdateKeyFrames();

  const bool IsKeyFrame(double time) const;

  void GetNearDistKeyScanID(PosCloud::Ptr key_pos_selected);

  void GetNearTimeKeyScanID(PosCloud::Ptr key_pos_selected,
                            const double cur_time) const;

  void TransformLiDARFeature(const LiDARFeature& lf_in,
                             const Eigen::Matrix4d& T_in_to_out,
                             LiDARFeature& lf_out) const;

  void DownsampleLiDARFeature(const LiDARFeature& lf_in,
                              LiDARFeature& lf_out) const;

  void ExtractSurroundFeatures(const double cur_time);

  void SetTargetMap(const LiDARFeature& feature_map);

  /// target : feature_map_ds
  bool FindCorrespondence(const LiDARFeature& lf_cur,
                          const LiDARFeature& lf_cur_in_M);

  void DownSampleCorrespondence();

 private:
  Trajectory::Ptr trajectory_;

  bool use_corner_feature_;

  /// current scan
  LiDARFeature feature_cur_;
  LiDARFeature feature_cur_ds_;
  int edge_min_valid_num_;
  int surf_min_valid_num_;
  float corner_leaf_size_;
  float surface_leaf_size_;

  /// correspondences of newest scan
  Eigen::aligned_vector<PointCorrespondence> point_correspondence_;
  Eigen::aligned_vector<PointCorrespondence> point_correspondence_ds_;
  /// downsample param of correspondences
  int cor_downsample_;

  bool update_full_cloud_;
  /// map
  LiDARFeature feature_map_;     // full_cloud is set by update_full_cloud_
  LiDARFeature feature_map_ds_;  // full_cloud is empty
  pcl::KdTreeFLANN<PosPoint>::Ptr kdtree_corner_map_;
  pcl::KdTreeFLANN<PosPoint>::Ptr kdtree_surface_map_;
  float keyframe_search_radius_;
  float keyframe_search_time_;
  float keyframe_density_map_;

  /// keyscan feature
  std::map<double, LiDARFeature> local_feature_container_;
  /// cached scan
  std::map<double, LiDARFeature> cache_feature_container_;
  /// 3D position of keyscan
  PosCloud::Ptr cloud_key_pos_;

  std::map<double, LiDARFeature> local_feature_container_all_ds_;

  // for loop closure
  PosCloud::Ptr cloud_key_pos_xy_;
  bool key_frame_updated_;

  /// Keyframe strategy
  double keyframe_angle_degree_;
  double keyframe_dist_meter_;
  double keyframe_time_second_;

  double cloud_reserved_time_;
};

}  // namespace clic
