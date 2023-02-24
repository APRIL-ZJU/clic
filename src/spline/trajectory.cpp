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

#include "trajectory.h"
#include <fstream>

namespace clic {

void Trajectory::GetIMUState(double time, IMUState &imu_state) const {
  SE3d pose = GetIMUPose(time);

  imu_state.timestamp = time;
  imu_state.q = pose.unit_quaternion();
  imu_state.p = pose.translation();
  imu_state.v = GetTransVelWorld(time);
  // imu_state.bias;
  // imu_state.g;
}

void Trajectory::UndistortScan(const PosCloud &scan_raw,
                               const double target_timestamp,
                               PosCloud &scan_in_target) const {
  scan_in_target.header = scan_raw.header;
  scan_in_target.resize(scan_raw.size());
  scan_in_target.is_dense = true;

  SE3d pose_G_to_target = GetLidarPose(target_timestamp).inverse();  // TL0_G

  std::size_t cnt = 0;
  for (auto const &raw_p : scan_raw.points) {
    if (pcl_isnan(raw_p.x)) {
      scan_in_target.is_dense = false;
      LOG(WARNING) << "[UndistortScan] input cloud exists NAN point";
      continue;
    }
    SE3d pose_Lk_to_G = GetLidarPose(raw_p.timestamp);  // TG_LK

    Eigen::Vector3d p_Lk(raw_p.x, raw_p.y, raw_p.z);
    Eigen::Vector3d point_out;
    point_out = pose_G_to_target * pose_Lk_to_G * p_Lk;

    PosPoint point;
    point.x = point_out(0);
    point.y = point_out(1);
    point.z = point_out(2);
    point.intensity = raw_p.intensity;
    point.timestamp = raw_p.timestamp;

    scan_in_target[cnt++] = point;
  }
}

void Trajectory::UndistortScanInG(const PosCloud &scan_raw,
                                  PosCloud &scan_in_target) const {
  scan_in_target.header = scan_raw.header;
  scan_in_target.resize(scan_raw.size());
  scan_in_target.is_dense = true;

  std::size_t cnt = 0;
  for (auto const &raw_p : scan_raw.points) {
    if (pcl_isnan(raw_p.x)) {
      scan_in_target.is_dense = false;
      LOG(WARNING) << "[UndistortScan] input cloud exists NAN point";
      continue;
    }
    SE3d pose_Lk_to_G = GetLidarPose(raw_p.timestamp);  // TG_LK

    Eigen::Vector3d p_Lk(raw_p.x, raw_p.y, raw_p.z);
    Eigen::Vector3d point_out;
    point_out = pose_Lk_to_G * p_Lk;

    PosPoint point;
    point.x = point_out(0);
    point.y = point_out(1);
    point.z = point_out(2);
    point.intensity = raw_p.intensity;
    point.timestamp = raw_p.timestamp;

    scan_in_target[cnt++] = point;
  }
}

SE3d Trajectory::GetSensorPose(const double timestamp,
                               const ExtrinsicParam &EP_StoI) const {
  double time_ns = timestamp * S_TO_NS + EP_StoI.t_offset_ns;

  if (!(time_ns >= this->minTimeNs() && time_ns < this->maxTimeNs())) {
    std::cout << time_ns << "; not in [" << this->minTimeNs() << ", "
              << this->maxTimeNs() << "]; "
              << "input time: " << timestamp
              << "[s]; t_offset: " << EP_StoI.t_offset_ns << " [ns]\n";
  }
  assert(time_ns >= this->minTimeNs() && time_ns < this->maxTimeNs() &&
         "[GetSensorPose] querry time not in range.");

  SE3d pose_I_to_G = this->poseNs(time_ns);
  SE3d pose_S_to_G = pose_I_to_G * EP_StoI.se3;
  return pose_S_to_G;
}

void Trajectory::ToTUMTxt(std::string traj_path, double dt) {
  std::ofstream outfile;
  outfile.open(traj_path);
  outfile.setf(std::ios::fixed);

  double min_time = minTime(IMUSensor);
  double max_time = maxTime(IMUSensor);
  for (double t = min_time; t < max_time; t += dt) {
    SE3d pose = GetIMUPose(t);
    Eigen::Vector3d p = pose.translation();
    Eigen::Quaterniond q = pose.unit_quaternion();

    /// uncomment this line for VIRAL dataset to align with gt,  extrinsic is from leica_prism.yaml
    p = (q.toRotationMatrix() * Eigen::Vector3d(-0.293656, -0.012288, -0.273095) + p).eval();

    double relative_bag_time = data_start_time_ + t;
    outfile.precision(9);
    outfile << relative_bag_time << " ";
    outfile.precision(5);
    outfile << p(0) << " " << p(1) << " " << p(2) << " " << q.x() << " "
            << q.y() << " " << q.z() << " " << q.w() << "\n";
  }
  outfile.close();
  std::cout << "Save trajectory at " << traj_path << std::endl;
}

}  // namespace clic
