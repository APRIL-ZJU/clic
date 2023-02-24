#pragma once

#include <algorithm>
#include <list>
#include <numeric>
#include <vector>

#include <ros/assert.h>
#include <ros/console.h>

#include "parameters.h"
#include "visual_struct.h"

namespace clic {

class FeatureManager {
 public:
  FeatureManager() {}

  void clearState();

  int getFeatureCount() const;

  bool addFeatureCheckParallax(
      int frame_count,
      const std::map<int, std::vector<std::pair<int, Vector7d>>>& image,
      double td);

  std::vector<std::pair<Vector3d, Vector3d>> getCorresponding(
      int frame_count_l, int frame_count_r,
      std::vector<int>* fea_ids = nullptr) const;

  // for ceree
  VectorXd getDepthVector() const;

  void setDepth(const VectorXd& x);

  // remove features with depth less than zero after BA
  void removeFailures();

  void clearDepth(const VectorXd& x);

  double triangulateOneFeature(const FeaturePerId& feature_checked,
                               const Eigen::Vector3d Ps_cam[],
                               const Eigen::Matrix3d Rs_cam[]) const;

  void triangulate(Vector3d Ps[], Matrix3d Rs[], double init_depth = -1);

  void removeBackShiftDepth(Matrix3d marg_R, Vector3d marg_P, Matrix3d new_R,
                            Vector3d new_P);

  void removeBack();

  void removeFront(int frame_count);

  static bool isLandmarkCandidate(const FeaturePerId& feature) {
    size_t used_num = feature.feature_per_frame.size();
    if (used_num >= 2 && feature.start_frame < WINDOW_SIZE - 2)
      return true;
    else
      return false;
  }

  std::list<FeaturePerId> feature;
  int last_track_num;

 private:

  double compensatedParallax2(const FeaturePerId& it_per_id, int frame_count);
};

}  // namespace clic