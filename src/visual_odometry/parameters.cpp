#include "parameters.h"
#include <opencv2/core.hpp>

#include <utils/opt_weight.h>

namespace clic {

double INIT_DEPTH;
double MIN_PARALLAX;

double FOCAL_LENGTH;

double ACC_N = 8.0e-2;
double GYR_N = 4.0e-3;
double ACC_W = 4.0e-5;
double GYR_W = 2.0e-6;

void readParameters(const YAML::Node& node) {
  /// IMU parameters
  IMUNoise imu_noise(node);
  ACC_N = imu_noise.sigma_a;
  GYR_N = imu_noise.sigma_w;
  ACC_W = imu_noise.sigma_ab;
  GYR_W = imu_noise.sigma_wb;

  std::string config_path = node["config_path"].as<std::string>();

  /// camera parameters
  std::string cam_yaml = config_path + node["camera_yaml"].as<std::string>();
  cv::FileStorage fsSettings(cam_yaml, cv::FileStorage::READ);
  if (!fsSettings.isOpened()) {
    std::cerr << "ERROR: Wrong path to settings" << std::endl;
  }

  FOCAL_LENGTH = fsSettings["focal_length"];

  MIN_PARALLAX = fsSettings["keyframe_parallax"];
  MIN_PARALLAX = MIN_PARALLAX / FOCAL_LENGTH;

  INIT_DEPTH = 20.0;

  fsSettings.release();
}

}  // namespace clic