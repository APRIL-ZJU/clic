#include "initial_sfm.h"

using namespace std;
using namespace Eigen;

// For a point in one frame; Note: P[3*4] = K*[R|t] and Pi[1*4]
// z * [u v 1]^T = P[3*4] * X[4*1];   ====>
// [u v 1]^T × P * X = 0;  ====>
// P0 - u*P2 = 0;
// P1 - v*P2 = 0;
// reference: https://zhuanlan.zhihu.com/p/63179478
// Pose0,Pose1 将3D点从世界坐标系转换至相机坐标系
// point0,point2 像素坐标
// return: point_3d 三角化得到的3d点
void GlobalSFM::triangulatePoint(const Eigen::Matrix<double, 3, 4> &Pose0,
                                 const Eigen::Matrix<double, 3, 4> &Pose1,
                                 const Vector2d &point0, const Vector2d &point1,
                                 Vector3d &point_3d) const {
  Matrix4d design_matrix = Matrix4d::Zero();
  design_matrix.row(0) = point0[0] * Pose0.row(2) - Pose0.row(0);
  design_matrix.row(1) = point0[1] * Pose0.row(2) - Pose0.row(1);
  design_matrix.row(2) = point1[0] * Pose1.row(2) - Pose1.row(0);
  design_matrix.row(3) = point1[1] * Pose1.row(2) - Pose1.row(1);
  Vector4d triangulated_point =
      design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();
  point_3d(0) = triangulated_point(0) / triangulated_point(3);
  point_3d(1) = triangulated_point(1) / triangulated_point(3);
  point_3d(2) = triangulated_point(2) / triangulated_point(3);
}

bool GlobalSFM::solveFrameByPnP(Matrix3d &R_initial, Vector3d &P_initial, int i,
                                vector<SFMFeature> &sfm_f) const {
  vector<cv::Point2f> pts_2_vector;
  vector<cv::Point3f> pts_3_vector;
  for (int j = 0; j < (int)sfm_f.size(); j++) {
    if (sfm_f[j].state != true) continue;
    Vector2d point2d;
    for (int k = 0; k < (int)sfm_f[j].observation.size(); k++) {
      if (sfm_f[j].observation[k].first == i) {
        Vector2d img_pts = sfm_f[j].observation[k].second;
        cv::Point2f pts_2(img_pts(0), img_pts(1));
        pts_2_vector.push_back(pts_2);
        cv::Point3f pts_3(sfm_f[j].position[0], sfm_f[j].position[1],
                          sfm_f[j].position[2]);
        pts_3_vector.push_back(pts_3);
        break;
      }
    }
  }
  if (int(pts_2_vector.size()) < 15) {
    printf("unstable features tracking, please slowly move you device!\n");
    if (int(pts_2_vector.size()) < 10) return false;
  }
  cv::Mat r, rvec, t, D, tmp_r;
  cv::eigen2cv(R_initial, tmp_r);
  cv::Rodrigues(tmp_r, rvec);
  cv::eigen2cv(P_initial, t);
  cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
  bool pnp_succ;
  pnp_succ = cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1);
  if (!pnp_succ) {
    return false;
  }
  cv::Rodrigues(rvec, r);
  // cout << "r " << endl << r << endl;
  MatrixXd R_pnp;
  cv::cv2eigen(r, R_pnp);
  MatrixXd T_pnp;
  cv::cv2eigen(t, T_pnp);
  R_initial = R_pnp;
  P_initial = T_pnp;
  return true;
}

void GlobalSFM::triangulateTwoFrames(const int frame0,
                                     const Eigen::Matrix<double, 3, 4> &Pose0,
                                     const int frame1,
                                     const Eigen::Matrix<double, 3, 4> &Pose1,
                                     vector<SFMFeature> &sfm_f) const {
  if (frame0 == frame1) {
    std::cout << "frame0 " << frame0 << "; frame1: " << frame1 << std::endl;
  }
  assert(frame0 != frame1);
  for (int j = 0; j < (int)sfm_f.size(); j++) {
    if (sfm_f[j].state == true) continue;
    bool has_0 = false, has_1 = false;
    Vector2d point0;
    Vector2d point1;

    for (int k = 0; k < (int)sfm_f[j].observation.size(); k++) {
      if (sfm_f[j].observation[k].first == frame0) {
        point0 = sfm_f[j].observation[k].second;
        has_0 = true;
      }
      if (sfm_f[j].observation[k].first == frame1) {
        point1 = sfm_f[j].observation[k].second;
        has_1 = true;
      }
    }
    if (has_0 && has_1) {
      Vector3d point_3d;
      triangulatePoint(Pose0, Pose1, point0, point1, point_3d);
      sfm_f[j].state = true;
      sfm_f[j].position = point_3d;
      // cout << "trangulated : " << frame1 << "  3d point : "  << j << "  " <<
      // point_3d.transpose() << endl;
    }
  }
}

bool GlobalSFM::construct(const int frame_num, const int ref_frame_idx,
                          const int cur_fixed_idx,
                          std::vector<SFMFeature> &sfm_f, Eigen::Vector3d Ps[],
                          Eigen::Matrix3d Rs[]) const {
  std::map<int, Eigen::Vector3d> sfm_tracked_points;

  assert(ref_frame_idx < cur_fixed_idx &&
         "ref frame idx large than current fixed frame.");

  Eigen::Matrix<double, 3, 4> Pose[frame_num];
  for (int i = 0; i <= cur_fixed_idx; i++) {
    Pose[i].block<3, 3>(0, 0) = Rs[i];
    Pose[i].block<3, 1>(0, 3) = Ps[i];
  }

  // 历史帧与 cur_fixed_idx 三角化
  for (int i = ref_frame_idx; i < cur_fixed_idx; i++) {
    triangulateTwoFrames(i, Pose[i], cur_fixed_idx, Pose[cur_fixed_idx], sfm_f);
  }
  // 历史帧与 ref_frame_idx 三角化
  for (int i = ref_frame_idx + 1; i < cur_fixed_idx; i++) {
    triangulateTwoFrames(ref_frame_idx, Pose[ref_frame_idx], i, Pose[i], sfm_f);
  }

  // 在 ref_frame_idx 之前的帧与 ref_frame_idx 三角化
  if (ref_frame_idx > 0) {
    for (int i = 0; i < ref_frame_idx; i++) {
      triangulateTwoFrames(ref_frame_idx, Pose[ref_frame_idx], i, Pose[i],
                           sfm_f);
    }
  }

  // 基于三角化恢复的3D点，初始化其他位姿
  if (cur_fixed_idx < frame_num) {
    for (int i = cur_fixed_idx + 1; i < frame_num; i++) {
      // solve pnp
      Matrix3d R_initial = Rs[i - 1];
      Vector3d P_initial = Ps[i - 1];
      if (!solveFrameByPnP(R_initial, P_initial, i, sfm_f)) return false;
      Rs[i] = R_initial;
      Ps[i] = P_initial;
      // triangulate
      triangulateTwoFrames(i, Pose[i], ref_frame_idx, Pose[ref_frame_idx],
                           sfm_f);
    }
  }

#if false
  // triangulate all other points
  for (int j = 0; j <(int) sfm_f.size(); j++) {
    if (sfm_f[j].state == true) continue;
    if ((int)sfm_f[j].observation.size() >= 2) {
      Vector2d point0, point1;
      int frame_0 = sfm_f[j].observation[0].first;
      point0 = sfm_f[j].observation[0].second;
      int frame_1 = sfm_f[j].observation.back().first;
      point1 = sfm_f[j].observation.back().second;
      Vector3d point_3d;
      triangulatePoint(Pose[frame_0], Pose[frame_1], point0, point1, point_3d);
      sfm_f[j].state = true;
      sfm_f[j].position = point_3d;
    }
  }
#endif
  // TODO: Full BA to opitimiaze the poses and landmarks
  return true;
}

bool GlobalSFM::construct_orignal(
    int frame_num, int l, const Eigen::Matrix3d relative_R,
    const Eigen::Vector3d relative_T, Eigen::Quaterniond q_out[],
    Eigen::Vector3d T_out[], std::vector<SFMFeature> &sfm_f,
    std::map<int, Eigen::Vector3d> &sfm_tracked_points) const {
  int feature_num = sfm_f.size();
  // 以第l帧为参考帧；第l、frame_num - 1 帧位姿已知
  // intial two view;
  q_out[l].setIdentity();
  T_out[l].setZero();
  q_out[frame_num - 1] = q_out[l] * Quaterniond(relative_R);
  T_out[frame_num - 1] = relative_T;

  // rotate to cam frame
  Matrix3d c_Rotation[frame_num];
  Vector3d c_Translation[frame_num];
  Eigen::Matrix<double, 3, 4> Pose[frame_num];

  std::vector<int> known_idx = {l, frame_num - 1};
  for (auto const &i : known_idx) {
    c_Rotation[i] = q_out[i].inverse().toRotationMatrix();
    c_Translation[i] = -1 * (c_Rotation[i] * T_out[i]);
    Pose[i].block<3, 3>(0, 0) = c_Rotation[i];
    Pose[i].block<3, 1>(0, 3) = c_Translation[i];
    // std::cout << "init " << i << "= p: " << T_out[i].transpose()
    //           << "; q: " << q_out[i].coeffs().transpose() << std::endl;
  }

  // 1: trangulate between l ----- frame_num - 1
  // 2: solve pnp l + 1; trangulate l + 1 ------- frame_num - 1;
  for (int i = l; i < frame_num - 1; i++) {
    // solve pnp
    if (i > l) {
      Matrix3d R_initial = c_Rotation[i - 1];
      Vector3d P_initial = c_Translation[i - 1];
      if (!solveFrameByPnP(R_initial, P_initial, i, sfm_f)) return false;
      c_Rotation[i] = R_initial;
      c_Translation[i] = P_initial;
      Pose[i].block<3, 3>(0, 0) = c_Rotation[i];
      Pose[i].block<3, 1>(0, 3) = c_Translation[i];
    }

    // triangulate point based on the solve pnp result
    triangulateTwoFrames(i, Pose[i], frame_num - 1, Pose[frame_num - 1], sfm_f);
  }
  // 3: triangulate l-----l+1 l+2 ... frame_num -2
  for (int i = l + 1; i < frame_num - 1; i++)
    triangulateTwoFrames(l, Pose[l], i, Pose[i], sfm_f);
  // 4: solve pnp l-1; triangulate l-1 ----- l
  //             l-2              l-2 ----- l
  for (int i = l - 1; i >= 0; i--) {
    // solve pnp
    Matrix3d R_initial = c_Rotation[i + 1];
    Vector3d P_initial = c_Translation[i + 1];
    if (!solveFrameByPnP(R_initial, P_initial, i, sfm_f)) return false;
    c_Rotation[i] = R_initial;
    c_Translation[i] = P_initial;
    Pose[i].block<3, 3>(0, 0) = c_Rotation[i];
    Pose[i].block<3, 1>(0, 3) = c_Translation[i];
    // triangulate
    triangulateTwoFrames(i, Pose[i], l, Pose[l], sfm_f);
  }
  // 5: triangulate all other points
  for (int j = 0; j < feature_num; j++) {
    if (sfm_f[j].state == true) continue;
    if ((int)sfm_f[j].observation.size() >= 2) {
      Vector2d point0, point1;
      int frame_0 = sfm_f[j].observation[0].first;
      point0 = sfm_f[j].observation[0].second;
      int frame_1 = sfm_f[j].observation.back().first;
      point1 = sfm_f[j].observation.back().second;
      Vector3d point_3d;
      triangulatePoint(Pose[frame_0], Pose[frame_1], point0, point1, point_3d);
      sfm_f[j].state = true;
      sfm_f[j].position = point_3d;
      // cout << "trangulated : " << frame_0 << " " << frame_1 << "  3d point :
      // "  << j << "  " << point_3d.transpose() << endl;
    }
  }

  // full BA
  ceres::Problem problem;
  ceres::LocalParameterization *local_parameterization =
      new ceres::QuaternionParameterization();

  double c_rotation[frame_num][4];
  for (int i = 0; i < frame_num; ++i) {
    // double array for ceres
    Quaterniond q(c_Rotation[i]);
    c_rotation[i][0] = q.w();
    c_rotation[i][1] = q.x();
    c_rotation[i][2] = q.y();
    c_rotation[i][3] = q.z();

    problem.AddParameterBlock(c_rotation[i], 4, local_parameterization);
    problem.AddParameterBlock(c_Translation[i].data(), 3);
    if (i == l) {
      problem.SetParameterBlockConstant(c_rotation[i]);
    }
    if (i == l || i == frame_num - 1) {
      problem.SetParameterBlockConstant(c_Translation[i].data());
    }
    // std::cout << i << "= p: " << c_Translation[i].transpose()
    //           << "; q: " << q.coeffs().transpose() << std::endl;
  }

  for (int i = 0; i < feature_num; i++) {
    if (sfm_f[i].state != true) continue;
    for (int j = 0; j < int(sfm_f[i].observation.size()); j++) {
      int frame_idx = sfm_f[i].observation[j].first;
      ceres::CostFunction *cost_function =
          ReprojectionError3D::Create(sfm_f[i].observation[j].second.x(),
                                      sfm_f[i].observation[j].second.y());

      problem.AddResidualBlock(cost_function, NULL, c_rotation[frame_idx],
                               c_Translation[frame_idx].data(),
                               sfm_f[i].position.data());
    }
  }
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  // options.minimizer_progress_to_stdout = true;
  options.max_solver_time_in_seconds = 0.2;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.BriefReport() << std::endl;
  if (summary.termination_type == ceres::CONVERGENCE ||
      summary.final_cost < 5e-03) {
    std::cout << "vision only BA converge" << std::endl;
  } else {
    std::cout << "vision only BA not converge " << std::endl;
    return false;
  }
  for (int i = 0; i < frame_num; i++) {
    q_out[i].w() = c_rotation[i][0];
    q_out[i].x() = c_rotation[i][1];
    q_out[i].y() = c_rotation[i][2];
    q_out[i].z() = c_rotation[i][3];
    q_out[i] = q_out[i].inverse();
    T_out[i] = -1 * (q_out[i] * c_Translation[i]);
  }

  for (int i = 0; i < (int)sfm_f.size(); i++) {
    if (sfm_f[i].state) {
      sfm_tracked_points[sfm_f[i].id] = sfm_f[i].position;
    }
  }
  return true;
}
