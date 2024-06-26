#include "vio_initial.h"

namespace clic {

void VIOInitialization::ProcessIMU(double timestamp,
                                   const Eigen::Vector3d& linear_acceleration,
                                   const Eigen::Vector3d& angular_velocity) {
  if (frame_count != 0) {
    double dt = timestamp - imu_timestamps_buf.back();
    tmp_pre_integration->push_back(dt, linear_acceleration, angular_velocity);
  }

  imu_timestamps_buf.push_back(timestamp);
  linear_acceleration_buf.push_back(linear_acceleration);
  angular_velocity_buf.push_back(angular_velocity);
}

void VIOInitialization::ProcessImage(
    const sensor_msgs::PointCloud::ConstPtr& img_msg,
    double traj_start_time = 0) {
  std::map<int, std::vector<std::pair<int, Vector7D>>> image;
  FeatureMsg2Image(img_msg, image);

  // update observation of festures
  if (f_manager.addFeatureCheckParallax(frame_count, image, t_offset_CtoI))
    slide_flag = SLIDE_OLD;
  else
    slide_flag = SLIDE_SECOND_NEW;

  // save data
  timestamps[frame_count] = img_msg->header.stamp.toSec() - traj_start_time;
  ImageFrame imageframe(image, timestamps[frame_count]);
  // the pre_integration of fist frame is nullptr
  imageframe.pre_integration = tmp_pre_integration;
  all_image_frame.insert(std::make_pair(timestamps[frame_count], imageframe));

  // for preintegration
  Eigen::Vector3d acc_0 = linear_acceleration_buf.back();
  Eigen::Vector3d gyr_0 = angular_velocity_buf.back();
  tmp_pre_integration =
      new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};

  if (frame_count == WINDOW_SIZE) {
    if ((timestamps[frame_count] - initial_timestamp) > 0.1) {
      initial_timestamp = timestamps[frame_count];
      initial_done = InitialStructure();
    }
    // if we success in initilization, reserve the current frame
    if (!initial_done) {
      SlideWindow();
    }
  } else
    frame_count++;
}

bool VIOInitialization::InitialStructure() {
  // check imu observibility
  {
    map<double, ImageFrame>::iterator frame_it;
    Eigen::Vector3d sum_g;
    for (frame_it = all_image_frame.begin(), frame_it++;
         frame_it != all_image_frame.end(); frame_it++) {
      double dt = frame_it->second.pre_integration->sum_dt;
      Eigen::Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
      sum_g += tmp_g;
    }

    Eigen::Vector3d aver_g;
    aver_g = sum_g * 1.0 / ((int)all_image_frame.size() - 1);
    double var = 0;
    for (frame_it = all_image_frame.begin(), frame_it++;
         frame_it != all_image_frame.end(); frame_it++) {
      double dt = frame_it->second.pre_integration->sum_dt;
      Eigen::Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
      var += (tmp_g - aver_g).transpose() * (tmp_g - aver_g);
    }
    var = sqrt(var / ((int)all_image_frame.size() - 1));
    if (var < 0.25) {
      ROS_INFO("IMU excitation not enouth!");
      // return false;
    }
  }

  // global sfm
  Quaterniond Q[frame_count + 1];
  Vector3d T[frame_count + 1];
  map<int, Vector3d> sfm_tracked_points;
  vector<SFMFeature> sfm_f;
  for (auto& it_per_id : f_manager.feature) {
    int imu_j = it_per_id.start_frame - 1;
    SFMFeature tmp_feature;
    tmp_feature.state = false;
    tmp_feature.id = it_per_id.feature_id;
    for (auto& it_per_frame : it_per_id.feature_per_frame) {
      imu_j++;
      Vector3d pts_j = it_per_frame.point;
      tmp_feature.observation.push_back(
          make_pair(imu_j, Eigen::Vector2d{pts_j.x(), pts_j.y()}));
    }
    sfm_f.push_back(tmp_feature);
  }
  Matrix3d relative_R;
  Vector3d relative_T;
  int l;
  if (!relativePose(relative_R, relative_T, l)) {
    return false;
  }

  GlobalSFM sfm;
  if (!sfm.construct_orignal(frame_count + 1, l, relative_R, relative_T, Q, T,
                             sfm_f, sfm_tracked_points)) {
    ROS_INFO("global SFM failed!");
    slide_flag = SLIDE_OLD;
    return false;
  }

  // solve pnp for all frame, including the non-keyframe in the window
  map<double, ImageFrame>::iterator frame_it;
  frame_it = all_image_frame.begin();
  for (int i = 0; frame_it != all_image_frame.end(); frame_it++) {
    // provide initial guess
    cv::Mat r, rvec, t, D, tmp_r;
    if ((frame_it->first) == timestamps[i]) {
      frame_it->second.is_key_frame = true;
      frame_it->second.R = Q[i].toRotationMatrix() * q_CtoI.inverse();
      frame_it->second.T = T[i];
      i++;
      continue;
    }
    
    if ((frame_it->first) > timestamps[i]) {
      i++;
    }

    Matrix3d R_inital = (Q[i].inverse()).toRotationMatrix();
    Vector3d P_inital = -R_inital * T[i];
    cv::eigen2cv(R_inital, tmp_r);
    cv::Rodrigues(tmp_r, rvec);
    cv::eigen2cv(P_inital, t);

    frame_it->second.is_key_frame = false;
    vector<cv::Point3f> pts_3_vector;
    vector<cv::Point2f> pts_2_vector;
    for (auto& id_pts : frame_it->second.points) {
      int feature_id = id_pts.first;
      for (auto& i_p : id_pts.second) {
        auto it = sfm_tracked_points.find(feature_id);
        if (it != sfm_tracked_points.end()) {
          Vector3d world_pts = it->second;
          cv::Point3f pts_3(world_pts(0), world_pts(1), world_pts(2));
          pts_3_vector.push_back(pts_3);
          Vector2d img_pts = i_p.second.head<2>();
          cv::Point2f pts_2(img_pts(0), img_pts(1));
          pts_2_vector.push_back(pts_2);
        }
      }
    }
    cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
    if (pts_3_vector.size() < 6) {
      cout << "pts_3_vector size " << pts_3_vector.size() << endl;
      ROS_INFO("Not enough points for solve pnp !");
      return false;
    }
    if (!cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1)) {
      ROS_INFO("solve pnp fail!");
      return false;
    }
    cv::Rodrigues(rvec, r);
    MatrixXd R_pnp, tmp_R_pnp;
    cv::cv2eigen(r, tmp_R_pnp);
    R_pnp = tmp_R_pnp.transpose();
    MatrixXd T_pnp;
    cv::cv2eigen(t, T_pnp);
    T_pnp = R_pnp * (-T_pnp);
    frame_it->second.R = R_pnp * q_CtoI.inverse();
    frame_it->second.T = T_pnp;
  }

  // all_image_frame
  if (VisualInitialAlign())
    return true;
  else {
    ROS_INFO("misalign visual structure with IMU");
    return false;
  }
}

bool VIOInitialization::VisualInitialAlign() {
  VisualIMUAlignment visual_imu_alignment(p_CinI);

  VectorXd x;
  // solve scale; non-keyframes are also used for initialization
  bool result = visual_imu_alignment.TryAlign(all_image_frame, Bgs, gravity, x);
  if (!result) {
    ROS_INFO("solve g failed!");
    return false;
  }

  // change state
  for (int i = 0; i <= frame_count; i++) {
    Matrix3d Ri = all_image_frame[timestamps[i]].R;
    Vector3d Pi = all_image_frame[timestamps[i]].T;
    Rs[i] = Ri;
    Ps[i] = Pi;
    all_image_frame[timestamps[i]].is_key_frame = true;
  }

  VectorXd dep = f_manager.getDepthVector();
  for (int i = 0; i < dep.size(); i++) dep[i] = -1;
  f_manager.clearDepth(dep);

  // triangulat on cam pose
  Eigen::Vector3d Ps_cam[(WINDOW_SIZE + 1)];
  Eigen::Matrix3d Rs_cam[(WINDOW_SIZE + 1)];
  for (int i = 0; i <= frame_count; i++) {
    Rs_cam[i] = Rs[i] * q_CtoI;
    Ps_cam[i] = Rs[i] * p_CinI + Ps[i];
  }
  f_manager.triangulate(Ps_cam, Rs_cam);

  double s = (x.tail<1>())(0);
  // for (int i = 0; i <= WINDOW_SIZE; i++) {
  //   pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
  // }
  for (int i = frame_count; i >= 0; i--)
    Ps[i] = s * Ps[i] - Rs[i] * p_CinI - (s * Ps[0] - Rs[0] * p_CinI);
  int kv = -1;
  map<double, ImageFrame>::iterator frame_i;
  for (frame_i = all_image_frame.begin(); frame_i != all_image_frame.end();
       frame_i++) {
    if (frame_i->second.is_key_frame) {
      kv++;
      Vs[kv] = frame_i->second.R * x.segment<3>(kv * 3);
    }
  }
  for (auto& it_per_id : f_manager.feature) {
    int used_num = it_per_id.feature_per_frame.size();
    if (!(used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2)) continue;
    it_per_id.estimated_depth *= s;
  }

  Matrix3d R0 = Utility::g2R(gravity);
  double yaw = Utility::R2ypr(R0 * Rs[0]).x();
  R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
  gravity = R0 * gravity;
  // Matrix3d rot_diff = R0 * Rs[0].transpose();
  Matrix3d rot_diff = R0;
  for (int i = 0; i <= frame_count; i++) {
    Ps[i] = rot_diff * Ps[i];
    Rs[i] = rot_diff * Rs[i];
    Vs[i] = rot_diff * Vs[i];
  }
  ROS_INFO_STREAM("g0     " << gravity.transpose());
  ROS_INFO_STREAM("my R0  " << Utility::R2ypr(Rs[0]).transpose());

  return true;
}

bool VIOInitialization::relativePose(Matrix3d& relative_R, Vector3d& relative_T,
                                     int& l) const {
  // find previous frame which contians enough correspondance and parallex with
  // newest frame
  bool has_enough_corres = false;
  bool has_enough_parallax = false;
  int cnt_corres = 0;
  double sum_parallax = 0;

  for (int i = 0; i < WINDOW_SIZE; i++) {
    vector<pair<Vector3d, Vector3d>> corres;
    corres = f_manager.getCorresponding(i, WINDOW_SIZE);
    cnt_corres += corres.size();
    if (corres.size() > 20) {
      has_enough_corres = true;

      double sum_parallax = 0;
      double average_parallax;
      for (int j = 0; j < int(corres.size()); j++) {
        Vector2d pts_0(corres[j].first(0), corres[j].first(1));
        Vector2d pts_1(corres[j].second(0), corres[j].second(1));
        double parallax = (pts_0 - pts_1).norm();
        sum_parallax = sum_parallax + parallax;
      }
      average_parallax = 1.0 * sum_parallax / int(corres.size());
      if (average_parallax * 460 > 30 &&
          m_estimator.solveRelativeRT(corres, relative_R, relative_T)) {
        l = i;
        ROS_INFO(
            "average_parallax %f choose l %d and newest frame to triangulate "
            "the whole structure",
            average_parallax * 460, l);
        return true;
      }

      sum_parallax += average_parallax * 460;

      if (average_parallax * 460 > 30) {
        has_enough_parallax = true;
      }
    }
  }

  if (!has_enough_corres) {
    ROS_INFO("Not enough correspondance feature [ave = %d]; Move device around",
             cnt_corres / WINDOW_SIZE);
  } else if (!has_enough_parallax) {
    ROS_INFO("Not enough parallax [ave = %.2f]; Move device around",
             sum_parallax / WINDOW_SIZE);
  }
  return false;
}

void VIOInitialization::SlideWindow() {
  // For debug
  if (false) {
    int frame_max = 0;
    for (const auto& it_per_id : f_manager.feature) {
      frame_max = std::max(frame_max, it_per_id.endFrame());
    }
    std::cout << "feature max cnt/frame_count: " << frame_max << "/"
              << frame_count << std::endl;
  }

  if (frame_count != WINDOW_SIZE) return;
  // delete images older than t0
  if (slide_flag == SLIDE_OLD) {
    // 1.delete the pre_integration
    map<double, ImageFrame>::iterator it_0 =
        all_image_frame.find(timestamps[0]);
    if (it_0->second.pre_integration != nullptr)
      delete it_0->second.pre_integration;
    it_0->second.pre_integration = nullptr;
    for (auto iter = all_image_frame.begin(); iter != it_0; ++iter) {
      if (iter->second.pre_integration) delete iter->second.pre_integration;
      iter->second.pre_integration = NULL;
    }
    // 2.delete the images
    all_image_frame.erase(all_image_frame.begin(), it_0);
    all_image_frame.erase(it_0);
  }

  // sliding window
  if (slide_flag == SLIDE_OLD) {
    for (int i = 0; i < WINDOW_SIZE; i++) {
      timestamps[i] = timestamps[i + 1];
    }
    timestamps[WINDOW_SIZE] = timestamps[WINDOW_SIZE - 1];
    f_manager.removeBack();
  } else {
    timestamps[WINDOW_SIZE - 1] = timestamps[WINDOW_SIZE];
    f_manager.removeFront(WINDOW_SIZE);
  }
}

void VIOInitialization::PublishSFMLandmarks() {
  if (pub_landmarks.getNumSubscribers() == 0) return;

  sensor_msgs::PointCloud landmarks;
  for (const auto& it_per_id : f_manager.feature) {
    int used_num = it_per_id.feature_per_frame.size();
    if (!(used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2)) continue;

    int imu_i = it_per_id.start_frame;
    Vector3d pts_i =
        it_per_id.feature_per_frame[0].point * it_per_id.estimated_depth;
    Vector3d w_pts_i = Rs[imu_i] * (q_CtoI * pts_i + p_CinI) + Ps[imu_i];

    geometry_msgs::Point32 p;
    p.x = w_pts_i(0);
    p.y = w_pts_i(1);
    p.z = w_pts_i(2);
    landmarks.points.push_back(p);

    // if (initial_done) {
    //   std::cout << it_per_id.feature_id << ": " << w_pts_i.transpose()
    //             << std::endl;
    // }
  }

  landmarks.header.stamp = ros::Time::now();
  landmarks.header.frame_id = "map";
  pub_landmarks.publish(landmarks);
}

}  // namespace clic