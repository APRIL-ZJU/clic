#pragma once

#include <execinfo.h>
#include <csignal>
#include <cstdio>
#include <iostream>
#include <queue>

#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>

#include "camera_models/CameraFactory.h"
#include "camera_models/CataCamera.h"
#include "camera_models/PinholeCamera.h"
#include "parameters.h"

#include <glog/logging.h>
#include <utils/tic_toc.h>

namespace feature_tracker {
bool inBorder(const cv::Point2f &pt);

void reduceVector(std::vector<cv::Point2f> &v, std::vector<uchar> status);
void reduceVector(std::vector<int> &v, std::vector<uchar> status);

class FeatureTracker {
 public:
  FeatureTracker() {}

  void readImage(const cv::Mat &_img, double _cur_time);

  void readIntrinsicParameter(const std::string &calib_file);

  bool updateID(unsigned int i);

  // visualize the undistorted features
  void showUndistortion(const std::string &name);

  cv::Mat mask;         
  cv::Mat fisheye_mask;

  /// Notice: prev_img, prev_pts are left unused
  double prev_time, cur_time;
  cv::Mat prev_img, cur_img, forw_img;  // raw image:
  std::vector<cv::Point2f> prev_pts, cur_pts, forw_pts;  // raw feature
  std::vector<cv::Point2f> prev_un_pts, cur_un_pts;  // undistorted and normalized feature
  std::map<int, cv::Point2f> prev_un_pts_map, cur_un_pts_map;  // <id, feature>
  std::vector<cv::Point2f> pts_velocity;  // velocity of feature
  std::vector<int> ids;                   // track id
  std::vector<int> track_cnt;             // track count of fesatures in forw_img

  camodocal::CameraPtr m_camera;  // camera model

  static int n_id;

 private:
  void applyMask();

  // update feature in forw_pts, ids(-1), track_cnt(1)
  void addPoints(std::vector<cv::Point2f> &n_pts);

  // delete features if tracking lost or rejectWithF
  void reducePoints(std::vector<uchar> &status);

  // reject outlier by F
  void rejectWithF();

  // undistort cur_pts and save to cur_un_pts, cur_un_pts_map
  // get feature velocity in prev_un_pts_map
  void undistortedPoints();
};
}  // namespace feature_tracker