#pragma once

#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <vector>

class MotionEstimator {
 public:
  bool solveRelativeRT(
      const std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> &corres,
      Eigen::Matrix3d &R, Eigen::Vector3d &T,
      std::vector<int> *inliners = nullptr) const;
};
