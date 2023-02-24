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

#include "visual_odometry.h"

namespace clic {
void VisualOdometry::ClearState() {
  for (int i = 0; i < WINDOW_SIZE + 1; i++) {
    timestamps_[i] = -1;
  }
  frame_count_ = 0;
}

void VisualOdometry::AddImageToWindow(
    const sensor_msgs::PointCloud::ConstPtr& img_msg) {
  std::map<int, std::vector<std::pair<int, Vector7D>>> image;
  FeatureMsg2Image(img_msg, image);

  // save data
  timestamps_[frame_count_] =
      img_msg->header.stamp.toSec() - trajectory_->GetDataStartTime();

  // update observation of festures
  double t_offset =
      trajectory_->GetSensorEP(CameraSensor).t_offset_ns * NS_TO_S;
  if (this->addFeatureCheckParallax(frame_count_, image, t_offset))
    marg_flag_ = MARGIN_OLD;
  else
    marg_flag_ = MARGIN_SECOND_NEW;

  LOG(INFO) << "this frame is " << (marg_flag_ ? "Non-keyframe" : "Keyframe")
            << "; number of feature: " << this->getFeatureCount()
            << "; time: " << timestamps_[frame_count_];
}

void VisualOdometry::SlideWindow() {
  // window not fill
  if (frame_count_ < WINDOW_SIZE) {
    //   cur pos has data
    if (timestamps_[frame_count_] >= 0) frame_count_++;
    return;
  }

  // 管理滑窗内的特征
  if (marg_flag_ == MARGIN_OLD) {
    marg_old_t0_ = timestamps_[0];
    for (int i = 0; i < WINDOW_SIZE; i++) {
      timestamps_[i] = timestamps_[i + 1];
    }
    timestamps_[WINDOW_SIZE] = timestamps_[WINDOW_SIZE - 1];
    SlideWindowOld();

  } else {
    timestamps_[WINDOW_SIZE - 1] = timestamps_[WINDOW_SIZE];
    SlideWindowNew();
  }
}

void VisualOdometry::SlideWindowNew() {
  this->removeFailures();
  this->removeFront(frame_count_);
}

void VisualOdometry::SlideWindowOld() {
  this->removeFailures();

#if true
  auto pose0 = trajectory_->GetCameraPose(marg_old_t0_);
  auto pose1 = trajectory_->GetCameraPose(timestamps_[0]);
  Eigen::Matrix3d R0 = pose0.so3().matrix();
  Eigen::Vector3d P0 = pose0.translation();
  Eigen::Matrix3d R1 = pose1.so3().matrix();
  Eigen::Vector3d P1 = pose1.translation();
  this->removeBackShiftDepth(R0, P0, R1, P1);
#else
  this->removeBack();  //直接将特征点的起始帧号向前移动
#endif
}

double VisualOdometry::GetReprojectionError(
    const FeaturePerId& feature_checked, const Eigen::Vector3d& p_inG,
    const Eigen::Vector3d Ps_cam[], const Eigen::Matrix3d Rs_cam[],
    std::pair<int, int> project_frame) const {
  double reproject_error = 0;
  int cnt = 0;
  for (int k = 0; k < (int)feature_checked.feature_per_frame.size(); ++k) {
    int idx_j = feature_checked.start_frame + k;

    // 不在统计范围内的帧
    if (idx_j < project_frame.first || idx_j > project_frame.second) continue;

    Eigen::Vector3d pts_j = feature_checked.feature_per_frame.at(k).point;
    Eigen::Vector3d pts_j_hat =
        Rs_cam[idx_j].transpose() * (p_inG - Ps_cam[idx_j]);

    double depth_j = pts_j_hat.z();
    reproject_error +=
        ((pts_j_hat / depth_j).head<2>() - pts_j.head<2>()).norm();
    cnt++;
  }
  if (cnt > 0)
    reproject_error /= cnt;
  else
    reproject_error = 1e9;

  // in normalized plane
  return reproject_error;
}

bool VisualOdometry::TriangulateLandmarks(bool has_new_traj) {
  good_feature_ids_.clear();
  if (!IsWindowOk()) return false;

  int check_idx = WINDOW_SIZE;
  for (; check_idx > 0; check_idx--) {
    if (timestamps_[check_idx] <= last_landmark_time) break;
  }

  // triangulat on cam pose
  Eigen::Vector3d Ps_cam[(WINDOW_SIZE + 1)];
  Eigen::Matrix3d Rs_cam[(WINDOW_SIZE + 1)];
  for (int i = 0; i <= WINDOW_SIZE; i++) {
    const auto& poseCinG = trajectory_->GetCameraPose(timestamps_[i]);
    Rs_cam[i] = poseCinG.so3().matrix();
    Ps_cam[i] = poseCinG.translation();
  }

  double init_depth = 0.02;
  this->triangulate(Ps_cam, Rs_cam, init_depth);

  int reject[2] = {0, 0};
  int accept[2] = {0, 0};
  std::vector<double> error_vec;
  for (auto& it_per_id : this->feature) {
    if (it_per_id.feature_per_frame.size() <= 3) continue;
    if (it_per_id.estimated_depth < init_depth + 0.01) continue;

    // if (it_per_id.estimated_depth >= INIT_DEPTH) {
    //   it_per_id.estimated_depth = -1;
    //   it_per_id.pinG = Eigen::Vector3d::Zero();
    //   continue;
    // }

    Eigen::Vector3d w_pts;
    double reproject_error;
    int idx_tmp = 0;
    {
      if (it_per_id.pinG.norm() > 1e-9) {
        w_pts = it_per_id.pinG;
        idx_tmp = 0;
      } else {
        int idx_i = it_per_id.start_frame;
        Eigen::Vector3d pts_i =
            it_per_id.feature_per_frame[0].point * it_per_id.estimated_depth;
        w_pts = Rs_cam[idx_i] * pts_i + Ps_cam[idx_i];
        idx_tmp = 1;
      }

      int check_max_idx = WINDOW_SIZE;
      if (has_new_traj) {
        check_max_idx = check_idx;
      }
      reproject_error = GetReprojectionError(it_per_id, w_pts, Ps_cam, Rs_cam,
                                             {0, check_max_idx});
      if (reproject_error * 460 > 10) {
        reject[idx_tmp]++;

        // if (it_per_id.start_frame == 0 && it_per_id.endFrame() ==
        // WINDOW_SIZE) {
        //   LOG(INFO) << "[TriangulateLandmarks] reject long tracked fea "
        //             << it_per_id.feature_id
        //             << (idx_tmp == 0 ? " old " : " new ")
        //             << reproject_error * 460;
        // }
        it_per_id.estimated_depth = -1;
        it_per_id.pinG = Eigen::Vector3d::Zero();
        continue;
      }
    }
    accept[idx_tmp]++;

    it_per_id.pinG = w_pts;

    if (it_per_id.endFrame() == WINDOW_SIZE) {
      good_feature_ids_.push_back(it_per_id.feature_id);
      error_vec.push_back(reproject_error * 460);
    }
  }

  if (!error_vec.empty()) {
    std::sort(error_vec.rbegin(), error_vec.rend());
    LOG(INFO) << "good_feature_size: " << good_feature_ids_.size()
              << "; max error: " << error_vec.back();
  }

  LOG(INFO) << "old reject/accept: " << reject[0] << "/" << accept[0]
            << "; new reject/accept: " << reject[1] << "/" << accept[1];

  last_landmark_time = timestamps_[WINDOW_SIZE];
  LOG(INFO) << "[TriangulateLandmarks] time cal/cur/traj_max: "
            << last_landmark_time << " / " << timestamps_[WINDOW_SIZE] << " / "
            << trajectory_->maxTime(CameraSensor)
            << "; check_old_candidate check_idx: " << check_idx;

  return good_feature_ids_.size() > 0;
}

void VisualOdometry::SetDepth(const std::map<int, double> fea_id_inv_depths) {
  Eigen::Vector3d Ps_cam[(WINDOW_SIZE + 1)];
  Eigen::Matrix3d Rs_cam[(WINDOW_SIZE + 1)];
  for (int i = 0; i <= WINDOW_SIZE; i++) {
    const auto& poseCinG = trajectory_->GetCameraPose(timestamps_[i]);
    Rs_cam[i] = poseCinG.so3().matrix();
    Ps_cam[i] = poseCinG.translation();
  }

  for (auto& it_per_id : this->feature) {
    if (fea_id_inv_depths.find(it_per_id.feature_id) == fea_id_inv_depths.end())
      continue;

    double depth_inv = fea_id_inv_depths.at(it_per_id.feature_id);

    it_per_id.estimated_depth = 1.0 / depth_inv;

    // int i = it_per_id.start_frame;
    // Eigen::Vector3d pts_i =
    //     it_per_id.feature_per_frame[0].point * it_per_id.estimated_depth;
    // it_per_id.pinG = Rs_cam[i] * pts_i + Ps_cam[i];

    if (it_per_id.estimated_depth < 0)
      it_per_id.solve_flag = SolveFail;
    else
      it_per_id.solve_flag = SovelSucc;
  }
}

bool VisualOdometry::GetCurOptLandmarks(VPointCloud& landmarks_init,
                                        VPointCloud& landmarks_opt) const {
  LOG(INFO) << "[GetCurOptLandmarks] landmark_cal / landmark_cur / traj_max: "
            << last_landmark_time << " / " << timestamps_[WINDOW_SIZE] << " / "
            << trajectory_->maxTime(CameraSensor) << "\n";

  Eigen::Vector3d Ps_cam[(WINDOW_SIZE + 1)];
  Eigen::Matrix3d Rs_cam[(WINDOW_SIZE + 1)];
  for (int i = 0; i <= WINDOW_SIZE; i++) {
    const auto& poseCinG = trajectory_->GetCameraPose(timestamps_[i]);
    Rs_cam[i] = poseCinG.so3().matrix();
    Ps_cam[i] = poseCinG.translation();
  }

  for (auto& it_per_id : this->feature) {
    int i = it_per_id.start_frame;

    if (it_per_id.estimated_depth < 0) continue;

    if (std::find(good_feature_ids_.begin(), good_feature_ids_.end(),
                  it_per_id.feature_id) == good_feature_ids_.end())
      continue;

    VPoint p;
    p.x = it_per_id.pinG(0);
    p.y = it_per_id.pinG(1);
    p.z = it_per_id.pinG(2);
    p.intensity = landmarks_init.size();
    landmarks_init.push_back(p);

    Eigen::Vector3d pts_i =
        it_per_id.feature_per_frame[0].point * it_per_id.estimated_depth;
    Eigen::Vector3d w_pts_i = Rs_cam[i] * pts_i + Ps_cam[i];

    VPoint p_opt;
    p_opt.x = w_pts_i(0);
    p_opt.y = w_pts_i(1);
    p_opt.z = w_pts_i(2);
    p_opt.intensity = landmarks_opt.size();
    landmarks_opt.push_back(p_opt);
  }

  LOG(INFO) << "landmarks opt/init/feature in window: " << landmarks_opt.size()
            << "/" << landmarks_init.size() << "/" << this->feature.size()
            << std::endl;

  // debug
  {
    int candidate_num = 0, excellent_num = 0;
    for (auto& it_per_id : this->feature) {
      if (it_per_id.endFrame() == WINDOW_SIZE) {
        if (it_per_id.feature_per_frame.size() > WINDOW_SIZE * 3. / 4.)
          candidate_num++;
        if (it_per_id.start_frame == 0) excellent_num++;
      }
    }
    LOG(INFO) << "feature excellent/candidated/used num: " << excellent_num
              << "/" << candidate_num << "/" << landmarks_opt.size();
  }

  return true;
}

}  // namespace clic
