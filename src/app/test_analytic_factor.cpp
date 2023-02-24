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

#include <iostream>

#include <estimator/factor/ceres_local_param.h>
#include <lidar_odometry/lidar_feature.h>
#include <spline/trajectory.h>

#include <estimator/factor/analytic_diff/image_feature_factor.h>
#include <estimator/factor/analytic_diff/lidar_feature_factor.h>
#include <estimator/factor/analytic_diff/trajectory_value_factor.h>

#include <estimator/factor/auto_diff/image_feature_factor.h>
#include <estimator/factor/auto_diff/lidar_feature_factor.h>
#include <estimator/factor/auto_diff/trajectory_relative_value_factor.h>
#include <estimator/factor/auto_diff/trajectory_value_factor.h>

#include <visual_odometry/integration_base.h>

using namespace clic;
using namespace std;

class FactorTest {
 public:
  FactorTest() {
    local_parameterization_ = new LieLocalParameterization<SO3d>();
    analytic_local_parameterization_ =
        new LieAnalyticLocalParameterization<SO3d>();
    // For gravity
    homo_vec_local_parameterization_ =
        new ceres::HomogeneousVectorParameterization(3);

    problem_options_.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
    problem_options_.local_parameterization_ownership =
        ceres::DO_NOT_TAKE_OWNERSHIP;

    solver_options_.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    solver_options_.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    solver_options_.minimizer_progress_to_stdout = true;
    solver_options_.update_state_every_iteration = true;
    solver_options_.max_num_iterations = 1;
    solver_options_.num_threads = 1;

    trajectory_ = std::make_shared<Trajectory>(0.1);
    trajectory_->genRandomTrajectory(8);

    GenerateData();

    cout << "trajectory control points\n";
    trajectory_->print_knots();
    cout << "trajectory max time: " << trajectory_->maxTime(LiDARSensor)
         << endl;
  }

  void TestLocalVelocityFactorAutoDiff(
      Eigen::aligned_map<double*, Eigen::MatrixXd>& jacobians,
      bool use_analytic_factor = false) {
    int64_t time_ns = local_velocity_.timestamp * S_TO_NS + t_offset_ns_;

    SplineMeta<SplineOrder> spline_meta;
    trajectory_->CaculateSplineMeta(
        {{time_ns - t_padding_ns_, time_ns + t_padding_ns_}}, spline_meta);
    auto* cost_function = auto_diff::IMULocalVelocityFactor::Create(
        time_ns, local_velocity_.p, spline_meta, w_imu_);

    /// add so3 knots
    for (size_t i = 0; i < spline_meta.NumParameters(); i++) {
      cost_function->AddParameterBlock(4);
    }
    /// add vec3 knots
    for (size_t i = 0; i < spline_meta.NumParameters(); i++) {
      cost_function->AddParameterBlock(3);
    }
    cost_function->AddParameterBlock(1);  // t_offset_ns_

    cost_function->SetNumResiduals(3);

    ceres::Problem problem(problem_options_);
    std::vector<double*> vec;
    AddControlPoints(spline_meta, vec, false, use_analytic_factor, problem);
    AddControlPoints(spline_meta, vec, true, use_analytic_factor, problem);
    vec.push_back(&t_offset_ns_);

    problem.AddResidualBlock(cost_function, NULL, vec);

    GetJacobian(vec, problem, cost_function->num_residuals(), jacobians);
  }

  void TestLocalVelocityFactorAnalytic(
      Eigen::aligned_map<double*, Eigen::MatrixXd>& jacobians,
      bool use_analytic_factor = true) {
    int64_t time_ns = local_velocity_.timestamp * S_TO_NS + t_offset_ns_;

    SplineMeta<SplineOrder> spline_meta;
    trajectory_->CaculateSplineMeta(
        {{time_ns - t_padding_ns_, time_ns + t_padding_ns_}}, spline_meta);

    using Functor = analytic_derivative::LocalVelocityFactor;
    ceres::CostFunction* cost_function = new Functor(
        time_ns, local_velocity_.p, spline_meta.segments.at(0), w_imu_);

    ceres::Problem problem(problem_options_);
    std::vector<double*> vec;
    AddControlPoints(spline_meta, vec, false, use_analytic_factor, problem);
    AddControlPoints(spline_meta, vec, true, use_analytic_factor, problem);
    vec.push_back(&t_offset_ns_);

    problem.AddResidualBlock(cost_function, NULL, vec);

    GetJacobian(vec, problem, cost_function->num_residuals(), jacobians);
  }

  void TestImageFeatureFactorAutoDiff(
      Eigen::aligned_map<double*, Eigen::MatrixXd>& jacobians,
      int selected_case = 0, bool use_analytic_factor = false) {
    int k = selected_case;
    int64_t ti_ns = ti_[k] * S_TO_NS;
    int64_t tj_ns = tj_[k] * S_TO_NS;

    int64_t ti_ns_corrected = ti_[k] * S_TO_NS + t_offset_ns_;
    int64_t tj_ns_corrected = tj_[k] * S_TO_NS + t_offset_ns_;
    SplineMeta<SplineOrder> spline_meta;
    trajectory_->CaculateSplineMeta(
        {{ti_ns_corrected - t_padding_ns_, ti_ns_corrected + t_padding_ns_},
         {tj_ns_corrected - t_padding_ns_, tj_ns_corrected + t_padding_ns_}},
        spline_meta);

    auto* cost_function = auto_diff::ImageFeatureFactor::Create(
        ti_ns, pi_[k], tj_ns, pj_[k], spline_meta);

    /// add so3 knots
    for (size_t i = 0; i < spline_meta.NumParameters(); i++) {
      cost_function->AddParameterBlock(4);
    }
    /// add vec3 knots
    for (size_t i = 0; i < spline_meta.NumParameters(); i++) {
      cost_function->AddParameterBlock(3);
    }
    cost_function->AddParameterBlock(1);  // depth inverse
    cost_function->AddParameterBlock(1);  // time offset

    cost_function->SetNumResiduals(2);

    ceres::Problem problem(problem_options_);
    std::vector<double*> vec;
    AddControlPoints(spline_meta, vec, false, use_analytic_factor, problem);
    AddControlPoints(spline_meta, vec, true, use_analytic_factor, problem);
    vec.emplace_back(&depth_inv_);
    vec.emplace_back(&t_offset_ns_);

    problem.AddResidualBlock(cost_function, NULL, vec);

    GetJacobian(vec, problem, cost_function->num_residuals(), jacobians);
  }

  void TestImageFeatureFactorAnalytic(
      Eigen::aligned_map<double*, Eigen::MatrixXd>& jacobians,
      int selected_case = 0, bool use_analytic_factor = true) {
    int k = selected_case;
    int64_t ti_ns = ti_[k] * S_TO_NS;
    int64_t tj_ns = tj_[k] * S_TO_NS;

    int64_t ti_ns_corrected = ti_[k] * S_TO_NS + t_offset_ns_;
    int64_t tj_ns_corrected = tj_[k] * S_TO_NS + t_offset_ns_;
    SplineMeta<SplineOrder> spline_meta;
    trajectory_->CaculateSplineMeta(
        {{ti_ns_corrected - t_padding_ns_, ti_ns_corrected + t_padding_ns_},
         {tj_ns_corrected - t_padding_ns_, tj_ns_corrected + t_padding_ns_}},
        spline_meta);

    using Functor = analytic_derivative::ImageFeatureFactor;
    ceres::CostFunction* cost_function =
        new Functor(ti_ns, pi_[k], tj_ns, pj_[k], spline_meta);

    ceres::Problem problem(problem_options_);
    std::vector<double*> vec;
    AddControlPoints(spline_meta, vec, false, use_analytic_factor, problem);
    AddControlPoints(spline_meta, vec, true, use_analytic_factor, problem);
    vec.emplace_back(&depth_inv_);
    vec.emplace_back(&t_offset_ns_);

    problem.AddResidualBlock(cost_function, NULL, vec);

    GetJacobian(vec, problem, cost_function->num_residuals(), jacobians);
  }

  void TestImage3D2DFactorAutoDiff(
      Eigen::aligned_map<double*, Eigen::MatrixXd>& jacobians,
      int selected_case = 0, bool use_analytic_factor = false) {
    int k = selected_case;
    int64_t ti_ns = ti_[k] * S_TO_NS;

    int64_t ti_ns_corrected = ti_[k] * S_TO_NS + t_offset_ns_;

    SplineMeta<SplineOrder> spline_meta;
    trajectory_->CaculateSplineMeta(
        {{ti_ns_corrected - t_padding_ns_, ti_ns_corrected + t_padding_ns_}},
        spline_meta);

    auto* cost_function =
        auto_diff::Image3D2DFactor::Create(ti_ns, pi_[k], spline_meta);

    /// add so3 knots
    for (size_t i = 0; i < spline_meta.NumParameters(); i++) {
      cost_function->AddParameterBlock(4);
    }
    /// add vec3 knots
    for (size_t i = 0; i < spline_meta.NumParameters(); i++) {
      cost_function->AddParameterBlock(3);
    }
    cost_function->AddParameterBlock(3);  // p_G
    cost_function->AddParameterBlock(1);  // time offset

    cost_function->SetNumResiduals(2);

    ceres::Problem problem(problem_options_);
    std::vector<double*> vec;
    AddControlPoints(spline_meta, vec, false, use_analytic_factor, problem);
    AddControlPoints(spline_meta, vec, true, use_analytic_factor, problem);
    vec.emplace_back(p_G_.data());
    vec.emplace_back(&t_offset_ns_);

    problem.AddResidualBlock(cost_function, NULL, vec);

    GetJacobian(vec, problem, cost_function->num_residuals(), jacobians);
  }

  void TestImage3D2DFactorAnalytic(
      Eigen::aligned_map<double*, Eigen::MatrixXd>& jacobians,
      int selected_case = 0, bool use_analytic_factor = true) {
    int k = selected_case;
    int64_t ti_ns = ti_[k] * S_TO_NS;

    int64_t ti_ns_corrected = ti_[k] * S_TO_NS + t_offset_ns_;

    SplineMeta<SplineOrder> spline_meta;
    trajectory_->CaculateSplineMeta(
        {{ti_ns_corrected - t_padding_ns_, ti_ns_corrected + t_padding_ns_}},
        spline_meta);

    using Functor = analytic_derivative::Image3D2DFactor;
    ceres::CostFunction* cost_function =
        new Functor(ti_ns, pi_[k], spline_meta);

    ceres::Problem problem(problem_options_);
    std::vector<double*> vec;
    AddControlPoints(spline_meta, vec, false, use_analytic_factor, problem);
    AddControlPoints(spline_meta, vec, true, use_analytic_factor, problem);
    vec.emplace_back(p_G_.data());
    vec.emplace_back(&t_offset_ns_);

    problem.AddResidualBlock(cost_function, NULL, vec);

    GetJacobian(vec, problem, cost_function->num_residuals(), jacobians);
  }

  void TestImageFeatureOnePoseFactorAutoDiff(
      Eigen::aligned_map<double*, Eigen::MatrixXd>& jacobians,
      int selected_case = 0, bool use_analytic_factor = false) {
    int k = selected_case;
    int64_t time_j_ns = tj_[k] * S_TO_NS;

    SplineMeta<SplineOrder> spline_meta;
    trajectory_->CaculateSplineMeta({{time_j_ns, time_j_ns}}, spline_meta);

    auto pose_Ii_to_G = trajectory_->GetIMUPose(ti_[k]);
    auto* cost_function = auto_diff::ImageFeatureOnePoseFactor::Create(
        pi_[k], pose_Ii_to_G.so3(), pose_Ii_to_G.translation(), time_j_ns,
        pj_[k], spline_meta);

    /// add so3 knots
    for (size_t i = 0; i < spline_meta.NumParameters(); i++) {
      cost_function->AddParameterBlock(4);
    }
    /// add vec3 knots
    for (size_t i = 0; i < spline_meta.NumParameters(); i++) {
      cost_function->AddParameterBlock(3);
    }
    cost_function->AddParameterBlock(1);  // depth inverse

    cost_function->SetNumResiduals(2);

    ceres::Problem problem(problem_options_);
    std::vector<double*> vec;
    AddControlPoints(spline_meta, vec, false, use_analytic_factor, problem);
    AddControlPoints(spline_meta, vec, true, use_analytic_factor, problem);
    vec.emplace_back(&depth_inv_);

    problem.AddResidualBlock(cost_function, NULL, vec);

    GetJacobian(vec, problem, cost_function->num_residuals(), jacobians);
  }

  void TestImageFeatureOnePoseFactorAnalytic(
      Eigen::aligned_map<double*, Eigen::MatrixXd>& jacobians,
      int selected_case = 0, bool use_analytic_factor = true) {
    int k = selected_case;
    int64_t time_j_ns = tj_[k] * S_TO_NS;

    SplineMeta<SplineOrder> spline_meta;
    trajectory_->CaculateSplineMeta({{time_j_ns, time_j_ns}}, spline_meta);

    auto pose_Ii_to_G = trajectory_->GetIMUPose(ti_[k]);

    using Functor = analytic_derivative::ImageFeatureOnePoseFactor;
    ceres::CostFunction* cost_function =
        new Functor(pi_[k], pose_Ii_to_G.so3(), pose_Ii_to_G.translation(),
                    time_j_ns, pj_[k], spline_meta);

    ceres::Problem problem(problem_options_);
    std::vector<double*> vec;
    AddControlPoints(spline_meta, vec, false, use_analytic_factor, problem);
    AddControlPoints(spline_meta, vec, true, use_analytic_factor, problem);
    vec.emplace_back(&depth_inv_);

    problem.AddResidualBlock(cost_function, NULL, vec);

    GetJacobian(vec, problem, cost_function->num_residuals(), jacobians);
  }

  void TestImageDepthFactorAutoDiff(
      Eigen::aligned_map<double*, Eigen::MatrixXd>& jacobians,
      int selected_case = 0, bool use_analytic_factor = false) {
    int k = selected_case;

    auto pose_Ci_to_G = trajectory_->GetCameraPose(ti_[k]);
    auto pose_Cj_to_G = trajectory_->GetCameraPose(tj_[k]);
    auto pose_CitoCj = pose_Cj_to_G.inverse() * pose_Ci_to_G;

    auto* cost_function = auto_diff::ImageDepthFactor::Create(
        pi_[k], pj_[k], pose_CitoCj.so3(), pose_CitoCj.translation());

    cost_function->AddParameterBlock(1);  // depth inverse
    cost_function->SetNumResiduals(2);

    ceres::Problem problem(problem_options_);
    std::vector<double*> vec;

    vec.emplace_back(&depth_inv_);

    problem.AddResidualBlock(cost_function, NULL, vec);

    GetJacobian(vec, problem, cost_function->num_residuals(), jacobians);
  }

  void TestImageDepthFactorAnalytic(
      Eigen::aligned_map<double*, Eigen::MatrixXd>& jacobians,
      int selected_case = 0, bool use_analytic_factor = true) {
    int k = selected_case;
    auto pose_Ci_to_G = trajectory_->GetCameraPose(ti_[k]);
    auto pose_Cj_to_G = trajectory_->GetCameraPose(tj_[k]);
    auto pose_CitoCj = pose_Cj_to_G.inverse() * pose_Ci_to_G;

    using Functor = analytic_derivative::ImageDepthFactor;
    ceres::CostFunction* cost_function = new Functor(
        pi_[k], pj_[k], pose_CitoCj.so3(), pose_CitoCj.translation());

    ceres::Problem problem(problem_options_);
    std::vector<double*> vec;
    vec.emplace_back(&depth_inv_);

    problem.AddResidualBlock(cost_function, NULL, vec);

    GetJacobian(vec, problem, cost_function->num_residuals(), jacobians);
  }

  void TestEpipolarFactorAutoDiff(
      Eigen::aligned_map<double*, Eigen::MatrixXd>& jacobians,
      bool use_analytic_factor = false) {
    int64_t time_i_ns = Epipolar_ti_ * S_TO_NS;
    SplineMeta<SplineOrder> spline_meta;
    trajectory_->CaculateSplineMeta({{time_i_ns, time_i_ns}}, spline_meta);
    auto* cost_function = auto_diff::EpipolarFactor::Create(
        time_i_ns, Epipolar_xi_, Epipolar_xk_, Epipolar_S_GtoCk_,
        Epipolar_p_CkinG_, spline_meta, Epipolar_weight_);

    /// add so3 knots
    for (size_t i = 0; i < spline_meta.NumParameters(); i++) {
      cost_function->AddParameterBlock(4);
    }
    /// add vec3 knots
    for (size_t i = 0; i < spline_meta.NumParameters(); i++) {
      cost_function->AddParameterBlock(3);
    }

    cost_function->SetNumResiduals(1);

    ceres::Problem problem(problem_options_);
    std::vector<double*> vec;
    AddControlPoints(spline_meta, vec, false, use_analytic_factor, problem);
    AddControlPoints(spline_meta, vec, true, use_analytic_factor, problem);

    problem.AddResidualBlock(cost_function, NULL, vec);

    GetJacobian(vec, problem, cost_function->num_residuals(), jacobians);
  }

  void TestEpipolarFactorAnalytic(
      Eigen::aligned_map<double*, Eigen::MatrixXd>& jacobians,
      int selected_case = 0, bool use_analytic_factor = true) {
    int64_t time_i_ns = Epipolar_ti_ * S_TO_NS;
    SplineMeta<SplineOrder> spline_meta;
    trajectory_->CaculateSplineMeta({{time_i_ns, time_i_ns}}, spline_meta);

    using Functor = analytic_derivative::EpipolarFactor;
    ceres::CostFunction* cost_function =
        new Functor(time_i_ns, Epipolar_xi_, Epipolar_xk_, Epipolar_S_GtoCk_,
                    Epipolar_p_CkinG_, spline_meta, Epipolar_weight_);

    ceres::Problem problem(problem_options_);
    std::vector<double*> vec;
    AddControlPoints(spline_meta, vec, false, use_analytic_factor, problem);
    AddControlPoints(spline_meta, vec, true, use_analytic_factor, problem);

    problem.AddResidualBlock(cost_function, NULL, vec);

    GetJacobian(vec, problem, cost_function->num_residuals(), jacobians);
  }

  void TestLoamFeatureFactorAutoDiff(
      Eigen::aligned_map<double*, Eigen::MatrixXd>& jacobians,
      bool use_analytic_factor = false) {
    int64_t time_map_ns = pc_corr_.t_map * S_TO_NS;
    int64_t time_point_ns = pc_corr_.t_point * S_TO_NS;
    SplineMeta<SplineOrder> spline_meta;
    trajectory_->CaculateSplineMeta(
        {{time_map_ns, time_map_ns}, {time_point_ns, time_point_ns}},
        spline_meta);

    auto* cost_function = auto_diff::PointFeatureFactor::Create(
        time_point_ns, time_map_ns, pc_corr_, spline_meta, w_loam_);

    /// add so3 knots
    for (size_t i = 0; i < spline_meta.NumParameters(); i++) {
      cost_function->AddParameterBlock(4);
    }
    /// add vec3 knots
    for (size_t i = 0; i < spline_meta.NumParameters(); i++) {
      cost_function->AddParameterBlock(3);
    }
    cost_function->AddParameterBlock(4);  // R_LtoI
    cost_function->AddParameterBlock(3);  // p_LinI
    cost_function->AddParameterBlock(1);  // time offset

    cost_function->SetNumResiduals(1);

    ceres::Problem problem(problem_options_);
    std::vector<double*> vec;
    AddControlPoints(spline_meta, vec, false, use_analytic_factor, problem);
    AddControlPoints(spline_meta, vec, true, use_analytic_factor, problem);
    vec.emplace_back(S_LtoI_.data());
    problem.AddParameterBlock(S_LtoI_.data(), 4, local_parameterization_);
    vec.emplace_back(p_LinI_.data());
    vec.emplace_back(&t_offset_);

    problem.AddResidualBlock(cost_function, NULL, vec);

    GetJacobian(vec, problem, cost_function->num_residuals(), jacobians);
  }

  void TestLoamFeatureFactorAnalytic(
      Eigen::aligned_map<double*, Eigen::MatrixXd>& jacobians,
      bool use_analytic_factor = true) {
    int64_t time_point_ns = pc_corr_.t_point * S_TO_NS;
    SplineMeta<SplineOrder> spline_meta;
    trajectory_->CaculateSplineMeta({{time_point_ns, time_point_ns}},
                                    spline_meta);

    SE3d T_MtoG = trajectory_->GetLidarPose(pc_corr_.t_map);
    SO3d S_GtoM = T_MtoG.so3().inverse();
    Eigen::Vector3d p_GinM = S_GtoM * (-T_MtoG.translation());

    using Functor = analytic_derivative::LoamFeatureFactor;
    ceres::CostFunction* cost_function =
        new Functor(time_point_ns, pc_corr_, spline_meta.segments.at(0), S_GtoM,
                    p_GinM, S_LtoI_, p_LinI_, w_loam_);

    ceres::Problem problem(problem_options_);
    std::vector<double*> vec;
    AddControlPoints(spline_meta, vec, false, use_analytic_factor, problem);
    AddControlPoints(spline_meta, vec, true, use_analytic_factor, problem);

    problem.AddResidualBlock(cost_function, NULL, vec);

    GetJacobian(vec, problem, cost_function->num_residuals(), jacobians);
  }

  void TestRalativeLoamFeatureFactorAnalytic(
      Eigen::aligned_map<double*, Eigen::MatrixXd>& jacobians,
      bool use_analytic_factor = true) {
    int64_t time_map_ns = pc_corr_.t_map * S_TO_NS;
    int64_t time_point_ns = pc_corr_.t_point * S_TO_NS;
    SplineMeta<SplineOrder> spline_meta;
    trajectory_->CaculateSplineMeta(
        {{time_map_ns, time_map_ns}, {time_point_ns, time_point_ns}},
        spline_meta);

    using Functor = analytic_derivative::RalativeLoamFeatureFactor;
    ceres::CostFunction* cost_function =
        new Functor(time_point_ns, time_map_ns, pc_corr_, spline_meta, w_loam_);

    ceres::Problem problem(problem_options_);
    std::vector<double*> vec;
    AddControlPoints(spline_meta, vec, false, use_analytic_factor, problem);
    AddControlPoints(spline_meta, vec, true, use_analytic_factor, problem);

    problem.AddResidualBlock(cost_function, NULL, vec);

    GetJacobian(vec, problem, cost_function->num_residuals(), jacobians);
  }

  void TestLoamFeatureOptMapPoseFactorAutoDiff(
      Eigen::aligned_map<double*, Eigen::MatrixXd>& jacobians,
      bool use_analytic_factor = false) {
    int64_t time_point_ns = pc_corr_.t_point * S_TO_NS;
    SplineMeta<SplineOrder> spline_meta;
    trajectory_->CaculateSplineMeta({{time_point_ns, time_point_ns}},
                                    spline_meta);
    auto* cost_function = auto_diff::LoamFeatureOptMapPoseFactor::Create(
        time_point_ns, pc_corr_, spline_meta, w_loam_);

    /// add so3 knots
    for (size_t i = 0; i < spline_meta.NumParameters(); i++) {
      cost_function->AddParameterBlock(4);
    }
    /// add vec3 knots
    for (size_t i = 0; i < spline_meta.NumParameters(); i++) {
      cost_function->AddParameterBlock(3);
    }
    cost_function->AddParameterBlock(4);  // map R
    cost_function->AddParameterBlock(3);  // map p

    cost_function->SetNumResiduals(1);

    ceres::Problem problem(problem_options_);
    std::vector<double*> vec;
    AddControlPoints(spline_meta, vec, false, use_analytic_factor, problem);
    AddControlPoints(spline_meta, vec, true, use_analytic_factor, problem);
    vec.emplace_back(S_ImtoG_.data());
    vec.emplace_back(p_IminG_.data());
    problem.AddParameterBlock(S_ImtoG_.data(), 4, local_parameterization_);

    problem.AddResidualBlock(cost_function, NULL, vec);

    GetJacobian(vec, problem, cost_function->num_residuals(), jacobians);
  }

  void TestLoamFeatureOptMapPoseFactorAnalytic(
      Eigen::aligned_map<double*, Eigen::MatrixXd>& jacobians,
      bool use_analytic_factor = true) {
    int64_t time_point_ns = pc_corr_.t_point * S_TO_NS;
    SplineMeta<SplineOrder> spline_meta;
    trajectory_->CaculateSplineMeta({{time_point_ns, time_point_ns}},
                                    spline_meta);

    using Functor = analytic_derivative::LoamFeatureOptMapPoseFactor;
    ceres::CostFunction* cost_function = new Functor(
        time_point_ns, pc_corr_, spline_meta.segments.at(0), w_loam_);

    ceres::Problem problem(problem_options_);
    std::vector<double*> vec;
    AddControlPoints(spline_meta, vec, false, use_analytic_factor, problem);
    AddControlPoints(spline_meta, vec, true, use_analytic_factor, problem);
    vec.emplace_back(S_ImtoG_.data());
    vec.emplace_back(p_IminG_.data());
    problem.AddParameterBlock(S_ImtoG_.data(), 4,
                              analytic_local_parameterization_);

    problem.AddResidualBlock(cost_function, NULL, vec);

    GetJacobian(vec, problem, cost_function->num_residuals(), jacobians);
  }

  void TestIMUPoseFactorAutoDiff(
      Eigen::aligned_map<double*, Eigen::MatrixXd>& jacobians,
      bool use_analytic_factor = false) {
    int64_t time_ns = pose_data_.timestamp * S_TO_NS + t_offset_ns_;
    SplineMeta<SplineOrder> spline_meta;
    trajectory_->CaculateSplineMeta(
        {{time_ns - t_padding_ns_, time_ns + t_padding_ns_}}, spline_meta);

    auto* cost_function = auto_diff::IMUPoseFactor::Create(
        time_ns, pose_data_, spline_meta, w_pose_, w_pose_);

    /// add so3 knots
    for (size_t i = 0; i < spline_meta.NumParameters(); i++) {
      cost_function->AddParameterBlock(4);
    }
    /// add vec3 knots
    for (size_t i = 0; i < spline_meta.NumParameters(); i++) {
      cost_function->AddParameterBlock(3);
    }
    cost_function->AddParameterBlock(1);  // t_offset_ns_

    cost_function->SetNumResiduals(6);

    ceres::Problem problem(problem_options_);
    std::vector<double*> vec;
    AddControlPoints(spline_meta, vec, false, use_analytic_factor, problem);
    AddControlPoints(spline_meta, vec, true, use_analytic_factor, problem);
    vec.push_back(&t_offset_ns_);

    problem.AddResidualBlock(cost_function, NULL, vec);

    GetJacobian(vec, problem, cost_function->num_residuals(), jacobians);
  }

  void TestIMUPoseFactorAnalytic(
      Eigen::aligned_map<double*, Eigen::MatrixXd>& jacobians,
      bool use_analytic_factor = true) {
    Eigen::Matrix<double, 6, 1> info_vec =
        w_pose_ * Eigen::Matrix<double, 6, 1>::Ones();

    int64_t time_ns = pose_data_.timestamp * S_TO_NS + t_offset_ns_;
    SplineMeta<SplineOrder> spline_meta;
    trajectory_->CaculateSplineMeta(
        {{time_ns - t_padding_ns_, time_ns + t_padding_ns_}}, spline_meta);

    ceres::CostFunction* cost_function = new analytic_derivative::IMUPoseFactor(
        time_ns, pose_data_, spline_meta.segments.at(0), info_vec);

    ceres::Problem problem(problem_options_);
    std::vector<double*> vec;
    AddControlPoints(spline_meta, vec, false, use_analytic_factor, problem);
    AddControlPoints(spline_meta, vec, true, use_analytic_factor, problem);
    vec.push_back(&t_offset_ns_);

    problem.AddResidualBlock(cost_function, NULL, vec);

    GetJacobian(vec, problem, cost_function->num_residuals(), jacobians);
  }

  void TestIMUFactorAutoDiff(
      Eigen::aligned_map<double*, Eigen::MatrixXd>& jacobians,
      bool use_analytic_factor = false) {
    int64_t time_ns = imu_data_.timestamp * S_TO_NS + t_offset_ns_;
    SplineMeta<SplineOrder> spline_meta;
    trajectory_->CaculateSplineMeta(
        {{time_ns - t_padding_ns_, time_ns + t_padding_ns_}}, spline_meta);

    auto* cost_function = auto_diff::IMUFactor::Create(
        time_ns, imu_data_, spline_meta, w_imu_, w_imu_);

    /// add SO3 knots
    for (size_t i = 0; i < spline_meta.NumParameters(); i++) {
      cost_function->AddParameterBlock(4);
    }
    /// add R3 knots
    for (size_t i = 0; i < spline_meta.NumParameters(); i++) {
      cost_function->AddParameterBlock(3);
    }
    cost_function->AddParameterBlock(3);  // gyro bias
    cost_function->AddParameterBlock(3);  // acce bias
    cost_function->AddParameterBlock(3);  // gravity
    cost_function->AddParameterBlock(1);  // t_offset_ns_

    cost_function->SetNumResiduals(6);

    ceres::Problem problem(problem_options_);
    std::vector<double*> vec;
    AddControlPoints(spline_meta, vec, false, use_analytic_factor, problem);
    AddControlPoints(spline_meta, vec, true, use_analytic_factor, problem);
    vec.emplace_back(imu_bias_.gyro_bias.data());
    vec.emplace_back(imu_bias_.accel_bias.data());
    vec.emplace_back(gravity_.data());
    vec.push_back(&t_offset_ns_);

    problem.AddParameterBlock(gravity_.data(), 3,
                              homo_vec_local_parameterization_);
    problem.AddResidualBlock(cost_function, NULL, vec);

    GetJacobian(vec, problem, cost_function->num_residuals(), jacobians);
  }

  void TestIMUFactorAnalytic(
      Eigen::aligned_map<double*, Eigen::MatrixXd>& jacobians,
      bool use_analytic_factor = true) {
    int64_t time_ns = imu_data_.timestamp * S_TO_NS + t_offset_ns_;
    SplineMeta<SplineOrder> spline_meta;
    trajectory_->CaculateSplineMeta(
        {{time_ns - t_padding_ns_, time_ns + t_padding_ns_}}, spline_meta);

    Eigen::Matrix<double, 6, 1> info_vec =
        w_imu_ * Eigen::Matrix<double, 6, 1>::Ones();
    ceres::CostFunction* cost_function = new analytic_derivative::IMUFactor(
        time_ns, imu_data_, spline_meta.segments.at(0), info_vec);

    ceres::Problem problem(problem_options_);
    std::vector<double*> vec;
    AddControlPoints(spline_meta, vec, false, use_analytic_factor, problem);
    AddControlPoints(spline_meta, vec, true, use_analytic_factor, problem);
    vec.emplace_back(imu_bias_.gyro_bias.data());
    vec.emplace_back(imu_bias_.accel_bias.data());
    vec.emplace_back(gravity_.data());
    vec.push_back(&t_offset_ns_);

    problem.AddParameterBlock(gravity_.data(), 3,
                              homo_vec_local_parameterization_);
    problem.AddResidualBlock(cost_function, NULL, vec);

    GetJacobian(vec, problem, cost_function->num_residuals(), jacobians);
  }

  void TestPreIntegrationFactorAutoDiff(
      Eigen::aligned_map<double*, Eigen::MatrixXd>& jacobians,
      bool use_analytic_factor = false) {
    int64_t time0_ns = t_pre_inte[0] * S_TO_NS;
    int64_t time1_ns = t_pre_inte[1] * S_TO_NS;
    SplineMeta<SplineOrder> spline_meta;
    SplineMeta<BiasSplineOrder> spline_meta_bias;
    trajectory_->CaculateSplineMeta({{time0_ns, time1_ns}}, spline_meta);

    auto* cost_function = auto_diff::PreIntegrationFactor::Create(
        pre_integration_, time0_ns, time1_ns, spline_meta);

    for (size_t i = 0; i < spline_meta.NumParameters(); i++) {
      cost_function->AddParameterBlock(4);  /// add SO3 knots
    }
    for (size_t i = 0; i < spline_meta.NumParameters(); i++) {
      cost_function->AddParameterBlock(3);  /// add R3 knots
    }
    cost_function->AddParameterBlock(3);
    cost_function->AddParameterBlock(3);
    cost_function->AddParameterBlock(3);
    cost_function->AddParameterBlock(3);

    cost_function->SetNumResiduals(15);

    ceres::Problem problem(problem_options_);
    std::vector<double*> vec;
    AddControlPoints(spline_meta, vec, false, use_analytic_factor, problem);
    AddControlPoints(spline_meta, vec, true, use_analytic_factor, problem);
    vec.emplace_back(imu_bias_.gyro_bias.data());
    vec.emplace_back(imu_bias2_.gyro_bias.data());
    vec.emplace_back(imu_bias_.accel_bias.data());
    vec.emplace_back(imu_bias2_.accel_bias.data());

    problem.AddResidualBlock(cost_function, NULL, vec);

    GetJacobian(vec, problem, cost_function->num_residuals(), jacobians);
  }

  void TestPreIntegrationFactorAnalytic(
      Eigen::aligned_map<double*, Eigen::MatrixXd>& jacobians,
      std::vector<double*>& vec, std::vector<std::string>& param_descri,
      bool use_analytic_factor = true) {
    int64_t time0_ns = t_pre_inte[0] * S_TO_NS;
    int64_t time1_ns = t_pre_inte[1] * S_TO_NS;
    SplineMeta<SplineOrder> spline_meta;
    SplineMeta<BiasSplineOrder> spline_meta_bias;
    trajectory_->CaculateSplineMeta({{time0_ns, time1_ns}}, spline_meta);

    ceres::CostFunction* cost_function =
        new analytic_derivative::PreIntegrationFactor(
            pre_integration_, time0_ns, time1_ns, spline_meta);

    ceres::Problem problem(problem_options_);

    AddControlPoints(spline_meta, vec, false, use_analytic_factor, problem);
    AddControlPoints(spline_meta, vec, true, use_analytic_factor, problem);
    vec.emplace_back(imu_bias_.gyro_bias.data());
    vec.emplace_back(imu_bias2_.gyro_bias.data());
    vec.emplace_back(imu_bias_.accel_bias.data());
    vec.emplace_back(imu_bias2_.accel_bias.data());

    problem.AddResidualBlock(cost_function, NULL, vec);

    GetJacobian(vec, problem, cost_function->num_residuals(), jacobians);

    for (int i = 0; i < (int)spline_meta.NumParameters(); i++)
      param_descri.push_back("rotation control point");
    for (int i = 0; i < (int)spline_meta.NumParameters(); i++)
      param_descri.push_back("position control point");

    param_descri.push_back("gyro bias control point");
    param_descri.push_back("gyro bias control point");
    param_descri.push_back("accel bias control point");
    param_descri.push_back("accel bias control point");
  }

  void TestRelativeRotFactorAutoDiff(
      Eigen::aligned_map<double*, Eigen::MatrixXd>& jacobians,
      int selected_case = 0, bool use_analytic_factor = false) {
    int k = selected_case;
    int64_t time_a_ns = ta_[k] * S_TO_NS;
    int64_t time_b_ns = tb_[k] * S_TO_NS;
    SplineMeta<SplineOrder> spline_meta;
    trajectory_->CaculateSplineMeta(
        {{time_a_ns, time_a_ns}, {time_b_ns, time_b_ns}}, spline_meta);

    auto* cost_function = auto_diff::RelativeOrientationFactor::Create(
        S_BtoA_, time_a_ns, time_b_ns, spline_meta);

    for (size_t i = 0; i < spline_meta.NumParameters(); i++) {
      cost_function->AddParameterBlock(4);  /// add SO3 knots
    }
    cost_function->SetNumResiduals(3);

    ceres::Problem problem(problem_options_);
    std::vector<double*> vec;
    AddControlPoints(spline_meta, vec, false, use_analytic_factor, problem);
    problem.AddResidualBlock(cost_function, NULL, vec);
    GetJacobian(vec, problem, cost_function->num_residuals(), jacobians);
  }

  void TestRelativeRotFactorAnalytic(
      Eigen::aligned_map<double*, Eigen::MatrixXd>& jacobians,
      int selected_case = 0, bool use_analytic_factor = true) {
    int k = selected_case;
    int64_t time_a_ns = ta_[k] * S_TO_NS;
    int64_t time_b_ns = tb_[k] * S_TO_NS;
    SplineMeta<SplineOrder> spline_meta;
    trajectory_->CaculateSplineMeta(
        {{time_a_ns, time_a_ns}, {time_b_ns, time_b_ns}}, spline_meta);

    Eigen::Vector3d sqrt_info_vec = Eigen::Vector3d(0.33, 0.27, 0.11);
    ceres::CostFunction* cost_function =
        new analytic_derivative::RelativeOrientationFactor(
            S_BtoA_, time_a_ns, time_b_ns, spline_meta, sqrt_info_vec);

    ceres::Problem problem(problem_options_);
    std::vector<double*> vec;
    AddControlPoints(spline_meta, vec, false, use_analytic_factor, problem);
    problem.AddResidualBlock(cost_function, NULL, vec);

    GetJacobian(vec, problem, cost_function->num_residuals(), jacobians);
  }

  void CheckJacobian(
      std::string factor_descri,
      Eigen::aligned_map<double*, Eigen::MatrixXd>& jacobs_automatic,
      Eigen::aligned_map<double*, Eigen::MatrixXd>& jacobs_analytic,
      const std::vector<double*>& parameters = {},
      const std::vector<std::string>& param_descri = {}) {
    bool check_pass = true;
    size_t cnt = 0;

    std::map<double*, int> parameters_map;
    if (!parameters.empty() && !param_descri.empty()) {
      for (int i = 0; i < (int)parameters.size(); ++i)
        parameters_map[parameters.at(i)] = i;
    }

    for (auto const& v : jacobs_analytic) {
      auto iter = jacobs_automatic.find(v.first);
      if (jacobs_automatic.end() != iter) {
        Eigen::MatrixXd diff = iter->second - v.second;
        if (diff.cwiseAbs().maxCoeff() > 1e-6) {
          // 按内存地址大小的距离，不是误差项中参数添加顺序的索引
          int idx;
          if (parameters.empty()) {
            idx = std::distance(jacobs_automatic.begin(), iter);
          } else {
            idx = parameters_map.at(iter->first);
          }

          std::cout << std::setiosflags(ios::fixed) << std::setw(15)
                    << std::setprecision(15);

          if (parameters.empty())
            cout << " ======== index " << idx << " ========\n";
          else
            cout << " ======== index " << idx << " " << param_descri.at(idx)
                 << " ========\n";
          cout << "auto diff\n"
               << iter->second << "\nanalytic:\n"
               << v.second << endl;
          check_pass = false;

          std::cout << std::setiosflags(ios::fixed) << std::setw(3)
                    << std::setprecision(3);
        } else {
          cnt++;
        }
      }
    }

    cout << factor_descri << " check [" << cnt << "/" << jacobs_analytic.size()
         << "] jacobians ok.\n\n";
    if (!check_pass) {
      cout << RED << factor_descri << " has some problems.\n" << RESET;
    }
  }

 private:
  void GenerateData() {
    Eigen::Vector3d sqrt_info_vec = Eigen::Vector3d(0.33, 0.27, 0.11);

    t_offset_ns_ = 1.e6;
    t_padding_ns_ = 1e8;  // 0.1s

    // relative orientation
    ta_[0] = 0.11;
    tb_[0] = 0.33;
    ta_[1] = 0.21;
    tb_[1] = 0.83;
    S_BtoA_ = Sophus::SO3<double>::exp(Eigen::Vector3d::Random() * M_PI);
    // analytic_derivative::RelativeOrientationFactor::sqrt_info.diagonal() =
    //     sqrt_info_vec;
    auto_diff::RelativeOrientationFactor::sqrt_info = Eigen::Matrix3d::Zero();
    auto_diff::RelativeOrientationFactor::sqrt_info.diagonal() = sqrt_info_vec;

    pre_integration_ = new IntegrationBase(
        Eigen::Vector3d(0.11, 0.12, -9.83), Eigen::Vector3d(0.1, 0.2, -0.1),
        Eigen::Vector3d(0.01, 0.02, -0.01), Eigen::Vector3d(0.01, 0.20, -0.01));
    pre_integration_->push_back(0.025, Eigen::Vector3d(0.1, 0.2, -9.83),
                                Eigen::Vector3d(0.1, 0.2, 0.3));
    pre_integration_->push_back(0.46, Eigen::Vector3d(0.6, 0.4, -9.83),
                                Eigen::Vector3d(0.3, 0.1, 0.17));
    t_pre_inte[0] = 0.21;
    t_pre_inte[1] = 0.695;

    // local velocity
    local_velocity_.timestamp = 0.23;
    local_velocity_.p = Eigen::Vector3d(0.2, 0.4, 0.5);

    ti_[0] = 0.13;
    tj_[0] = 0.34;
    pi_[0] = Eigen::Vector3d(1, 2, 1);
    pj_[0] = Eigen::Vector3d(4, 5, 1);

    ti_[1] = 0.11;
    tj_[1] = 0.63;
    pi_[1] = Eigen::Vector3d(4, 5, 1);
    pj_[1] = Eigen::Vector3d(1, 2, 1);

    depth_inv_ = 0.51;
    p_G_ = Eigen::Vector3d(3.1, 2.8, 1.3);
    // S_CtoI_ = Sophus::SO3<double>::exp(Eigen::Vector3d::Zero() * M_PI);
    S_CtoI_ = Sophus::SO3<double>::exp(Eigen::Vector3d::Random() * M_PI);
    p_CinI_ = Eigen::Vector3d(0.1, 0.2, 0.3);  // Eigen::Vector3d::Zero();//

    analytic_derivative::ImageFeatureFactor::SetParam(S_CtoI_, p_CinI_);
    auto_diff::ImageFeatureFactor::SetParam(S_CtoI_, p_CinI_);

    analytic_derivative::Image3D2DFactor::SetParam(S_CtoI_, p_CinI_);
    auto_diff::Image3D2DFactor::SetParam(S_CtoI_, p_CinI_);

    analytic_derivative::ImageFeatureOnePoseFactor::SetParam(S_CtoI_, p_CinI_);
    auto_diff::ImageFeatureOnePoseFactor::SetParam(S_CtoI_, p_CinI_);

    ExtrinsicParam EP_CtoI;
    EP_CtoI.p = p_CinI_;
    EP_CtoI.so3 = S_CtoI_;
    EP_CtoI.q = S_CtoI_.unit_quaternion();
    EP_CtoI.se3.so3() = S_CtoI_;
    EP_CtoI.se3.translation() = p_CinI_;
    EP_CtoI.t_offset_ns = 0;
    trajectory_->SetSensorExtrinsics(SensorType::CameraSensor, EP_CtoI);

    // Epipolar factor
    Epipolar_ti_ = 0.53;
    Epipolar_xi_ = Eigen::Vector3d(0.1, 0.2, 1);
    Epipolar_xk_ = Eigen::Vector3d(1.3, 1.2, 1);
    Epipolar_S_GtoCk_ =
        Sophus::SO3<double>::exp(Eigen::Vector3d::Random() * M_PI);
    Epipolar_p_CkinG_ = Eigen::Vector3d(1.3, 1.2, 1);
    Epipolar_weight_ = 0.11;

    analytic_derivative::EpipolarFactor::SetParam(S_CtoI_, p_CinI_);
    auto_diff::EpipolarFactor::SetParam(S_CtoI_, p_CinI_);

    pc_corr_.geo_type = Plane;
    pc_corr_.t_point = 0.55;
    pc_corr_.t_map = 0;
    pc_corr_.point = Eigen::Vector3d(1, 2, 3);
    pc_corr_.geo_plane = Eigen::Vector4d(1, 2, 3, 4).normalized();

    pc_corr_.geo_point = Eigen::Vector3d(1, 2, 3);
    pc_corr_.geo_normal = Eigen::Vector3d(3, 2, 1).normalized();

    pose_data_.timestamp = 0.55;
    S_LtoI_ = Sophus::SO3<double>::exp(Eigen::Vector3d::Random() * M_PI);
    p_LinI_ = Eigen::Vector3d(0.15, 0.52, 0.35);
    t_offset_ = 0.0;

    // LoamFeatureOptMapPoseFactor
    SE3d T_IMtoG = trajectory_->GetIMUPose(0.45);
    S_ImtoG_ = T_IMtoG.so3();
    p_IminG_ = T_IMtoG.translation();
    analytic_derivative::LoamFeatureOptMapPoseFactor::SetParam(S_LtoI_,
                                                               p_LinI_);
    analytic_derivative::RalativeLoamFeatureFactor::SetParam(S_LtoI_, p_LinI_);
    auto_diff::LoamFeatureOptMapPoseFactor::SetParam(S_LtoI_, p_LinI_);

    ExtrinsicParam EP_LtoI;
    EP_LtoI.p = p_LinI_;
    EP_LtoI.so3 = S_LtoI_;
    EP_LtoI.q = S_LtoI_.unit_quaternion();
    EP_LtoI.se3.so3() = S_LtoI_;
    EP_LtoI.se3.translation() = p_LinI_;
    EP_LtoI.t_offset_ns = t_offset_;
    trajectory_->SetSensorExtrinsics(SensorType::LiDARSensor, EP_LtoI);

    imu_data_.timestamp = 0.55;
    imu_data_.gyro = Eigen::Vector3d(0.1, 0.2, 0.3);
    imu_data_.accel = Eigen::Vector3d(0.1, 0.2, -9.8);
    imu_bias_.gyro_bias = Eigen::Vector3d(0.1, 0.2, 0.3);
    imu_bias_.accel_bias = Eigen::Vector3d(0.1, 0.2, 0.3);
    imu_bias2_.gyro_bias = Eigen::Vector3d(0.11, 0.21, 0.31);
    imu_bias2_.accel_bias = Eigen::Vector3d(0.11, 0.21, 0.31);

    gravity_ = Eigen::Vector3d(0, 0, 9.8);
    g_refine_ = gravity_factor::recover_gravity_param(gravity_);

    w_loam_ = 1.3;
    w_pose_ = 2.3;
    w_imu_ = 3.4;
  }

  void AddControlPoints(const SplineMeta<SplineOrder>& spline_meta,
                        std::vector<double*>& vec, bool addPosKnot,
                        bool use_analytic_factor, ceres::Problem& problem) {
    for (auto const& seg : spline_meta.segments) {
      size_t start_idx = trajectory_->GetCtrlIndex(seg.t0_ns);
      for (size_t i = start_idx; i < (start_idx + seg.NumParameters()); ++i) {
        if (addPosKnot) {
          vec.emplace_back(trajectory_->getKnotPos(i).data());
          problem.AddParameterBlock(trajectory_->getKnotPos(i).data(), 3);
        } else {
          vec.emplace_back(trajectory_->getKnotSO3(i).data());
          if (use_analytic_factor) {
            problem.AddParameterBlock(trajectory_->getKnotSO3(i).data(), 4,
                                      analytic_local_parameterization_);
          } else {
            problem.AddParameterBlock(trajectory_->getKnotSO3(i).data(), 4,
                                      local_parameterization_);
          }
        }
      }
    }
  }

  void GetJacobian(std::vector<double*> param_vec, ceres::Problem& problem,
                   int residual_num,
                   Eigen::aligned_map<double*, Eigen::MatrixXd>& jacobians) {
    double cost = 0.0;
    ceres::CRSMatrix J;
    std::vector<double> residuals;
    problem.Evaluate(ceres::Problem::EvaluateOptions(), &cost, &residuals,
                     nullptr, &J);

    Eigen::MatrixXd dense_jacobian(J.num_rows, J.num_cols);
    dense_jacobian.setZero();
    for (int r = 0; r < J.num_rows; ++r) {
      for (int idx = J.rows[r]; idx < J.rows[r + 1]; ++idx) {
        const int c = J.cols[idx];
        dense_jacobian(r, c) = J.values[idx];
      }
    }

    int cnt = 0;
    std::string right_descri = ")= ";
    if (residual_num > 1) right_descri += "\n";
    for (size_t i = 0; i < param_vec.size(); i++) {
      int local_size = problem.ParameterBlockLocalSize(param_vec.at(i));
      Eigen::MatrixXd jacob = Eigen::MatrixXd::Zero(residual_num, local_size);
      jacob = dense_jacobian.block(0, cnt, residual_num, local_size);
      cnt += local_size;

      jacobians[param_vec.at(i)] = jacob;
      // jacobians[param_vec.at(i)] = jacob;
      // cout << "J(" << std::setw(2) << i << right_descri << jacob <<
      // std::endl;
    }

    std::cout << "cost = " << cost << "; redisual: ";
    for (auto& r : residuals) std::cout << r << ", ";
    std::cout << "\n";

    // std::cout << "J = (" << J.num_rows << ", " << J.num_cols
    //           << ") with non - zero value; \n ";
    // for (int i = 0; i < J.num_rows; i++) {
    //   for (int j = J.rows[i]; j < J.rows[i + 1]; j++) {
    //     std::cout << "J(" << std::setw(2) << i << "," << std::setw(2)
    //               << J.cols[j] << ") = " << std::setw(10)
    //               << std::setiosflags(ios::fixed) << std::setprecision(3)
    //               << J.values[j] << "; ";
    //   }
    //   cout << endl;
    // }
  }

  void GetJacobian(std::vector<double*> param_vec,
                   const ceres::CostFunction* cost_function) {
    int num_residuals = cost_function->num_residuals();
    Eigen::MatrixXd residuals;
    residuals.setZero(num_residuals, 1);

    std::vector<double*> J_vec;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        Jacob[cost_function->parameter_block_sizes().size()];
    size_t cnt = 0;
    for (auto const v : cost_function->parameter_block_sizes()) {
      Jacob[cnt].setZero(num_residuals, v);
      J_vec.emplace_back(Jacob[cnt++].data());
    }

    cost_function->Evaluate(param_vec.data(), residuals.data(), J_vec.data());
    cout << "residuals = " << residuals.transpose() << endl;

    for (size_t i = 0; i < J_vec.size(); ++i) {
      if (num_residuals == 1)
        cout << "J[" << i << "] = " << Jacob[i] << endl;
      else
        cout << "J[" << i << "] = \n" << Jacob[i] << endl;
    }
  }

 public:
  Trajectory::Ptr trajectory_;

  double t_offset_ns_;
  int64_t t_padding_ns_;

  double ta_[2], tb_[2];
  SO3d S_BtoA_;

  // PreIntegration factor
  double t_pre_inte[2];
  IntegrationBase* pre_integration_;

  // local velocity factor
  VecData local_velocity_;

  // image projection factor
  double ti_[2], tj_[2];
  Eigen::Vector3d pi_[2], pj_[2];
  double depth_inv_;
  Eigen::Vector3d p_G_;
  Sophus::SO3<double> S_CtoI_;
  Eigen::Vector3d p_CinI_;

  // EpipolarFactor
  double Epipolar_ti_;
  Eigen::Vector3d Epipolar_xi_;
  Eigen::Vector3d Epipolar_xk_;
  SO3d Epipolar_S_GtoCk_;
  Eigen::Vector3d Epipolar_p_CkinG_;
  double Epipolar_weight_;

  // lidar factor
  PointCorrespondence pc_corr_;
  PoseData pose_data_;
  Sophus::SO3<double> S_LtoI_;
  Eigen::Vector3d p_LinI_;
  double t_offset_;

  // LoamFeatureOptMapPoseFactor
  Sophus::SO3<double> S_ImtoG_;
  Eigen::Vector3d p_IminG_;

  // IMU data factor
  IMUData imu_data_;
  IMUBias imu_bias_;
  IMUBias imu_bias2_;
  Eigen::Vector3d gravity_;
  Eigen::Vector2d g_refine_;

  double w_loam_, w_pose_, w_imu_;

  ceres::LocalParameterization* local_parameterization_;
  ceres::LocalParameterization* analytic_local_parameterization_;
  ceres::HomogeneousVectorParameterization* homo_vec_local_parameterization_;
  ceres::Problem::Options problem_options_;
  ceres::Solver::Options solver_options_;
};

int main(int argc, char** argv) {
  FactorTest factor_test;

  std::cout << std::setiosflags(ios::fixed) << std::setprecision(3);
  cout << "\n ===== TEST ===== \n\n";

  for (int i = 0; i < 2; ++i) {
    Eigen::aligned_map<double*, Eigen::MatrixXd> jacobs_automatic;
    Eigen::aligned_map<double*, Eigen::MatrixXd> jacobs_analytic;

    std::string descri =
        "relative orientation factor [case " + std::to_string(i) + "]";

    factor_test.TestRelativeRotFactorAutoDiff(jacobs_automatic, i);
    factor_test.TestRelativeRotFactorAnalytic(jacobs_analytic, i);
    factor_test.CheckJacobian(descri, jacobs_automatic, jacobs_analytic);
  }

  double ti[2] = {0.21, 0.21};
  double tj[2] = {0.35, 0.695};
  for (int i = 0; i < 2; ++i) {
    factor_test.t_pre_inte[0] = ti[i];
    factor_test.t_pre_inte[1] = tj[i];

    Eigen::aligned_map<double*, Eigen::MatrixXd> jacobs_automatic;
    Eigen::aligned_map<double*, Eigen::MatrixXd> jacobs_analytic;
    std::vector<double*> param_vec;
    std::vector<std::string> param_descri;

    std::string descri =
        "PreIntegration factor [case " + std::to_string(i) + "]";
    factor_test.TestPreIntegrationFactorAutoDiff(jacobs_automatic);
    factor_test.TestPreIntegrationFactorAnalytic(jacobs_analytic, param_vec,
                                                 param_descri);
    factor_test.CheckJacobian(descri, jacobs_automatic, jacobs_analytic,
                              param_vec, param_descri);
  }

  {
    Eigen::aligned_map<double*, Eigen::MatrixXd> jacobs_automatic;
    Eigen::aligned_map<double*, Eigen::MatrixXd> jacobs_analytic;

    factor_test.TestLocalVelocityFactorAutoDiff(jacobs_automatic);
    factor_test.TestLocalVelocityFactorAnalytic(jacobs_analytic);
    factor_test.CheckJacobian("Local velocity factor", jacobs_automatic,
                              jacobs_analytic);
  }

  for (int i = 0; i < 2; ++i) {
    Eigen::aligned_map<double*, Eigen::MatrixXd> jacobs_automatic;
    Eigen::aligned_map<double*, Eigen::MatrixXd> jacobs_analytic;

    std::string descri =
        "image feature factor [case " + std::to_string(i) + "]";
    factor_test.TestImageFeatureFactorAutoDiff(jacobs_automatic, i);
    factor_test.TestImageFeatureFactorAnalytic(jacobs_analytic, i);
    factor_test.CheckJacobian(descri, jacobs_automatic, jacobs_analytic);

    jacobs_automatic.clear();
    jacobs_analytic.clear();
    descri = "image 3D-2D factor [case " + std::to_string(i) + "]";
    factor_test.TestImage3D2DFactorAutoDiff(jacobs_automatic, i);
    factor_test.TestImage3D2DFactorAnalytic(jacobs_analytic, i);
    factor_test.CheckJacobian(descri, jacobs_automatic, jacobs_analytic);

    jacobs_automatic.clear();
    jacobs_analytic.clear();
    descri = "ImageFeatureOnePoseFactor [case " + std::to_string(i) + "]";
    factor_test.TestImageFeatureOnePoseFactorAutoDiff(jacobs_automatic, i);
    factor_test.TestImageFeatureOnePoseFactorAnalytic(jacobs_analytic, i);
    factor_test.CheckJacobian(descri, jacobs_automatic, jacobs_analytic);

    jacobs_automatic.clear();
    jacobs_analytic.clear();
    descri = "ImageDepthFactor [case " + std::to_string(i) + "]";
    factor_test.TestImageDepthFactorAutoDiff(jacobs_automatic, i);
    factor_test.TestImageDepthFactorAnalytic(jacobs_analytic, i);
    factor_test.CheckJacobian(descri, jacobs_automatic, jacobs_analytic);
  }

  {
    Eigen::aligned_map<double*, Eigen::MatrixXd> jacobs_automatic;
    Eigen::aligned_map<double*, Eigen::MatrixXd> jacobs_analytic;

    factor_test.TestEpipolarFactorAutoDiff(jacobs_automatic);
    factor_test.TestEpipolarFactorAnalytic(jacobs_analytic);
    factor_test.CheckJacobian("Epipolar Factor", jacobs_automatic,
                              jacobs_analytic);
  }

  for (int i = 0; i < 2; ++i) {
    Eigen::aligned_map<double*, Eigen::MatrixXd> jacobs_automatic;
    Eigen::aligned_map<double*, Eigen::MatrixXd> jacobs_analytic;

    std::string descri;
    if (0 == i) {
      factor_test.pc_corr_.t_point = 0.55;
      factor_test.pc_corr_.t_map = 0;

      descri = "[Plane]";
      factor_test.pc_corr_.geo_type = Plane;
    } else {
      descri = "[Line]";
      factor_test.pc_corr_.geo_type = Line;
    }
    factor_test.TestLoamFeatureFactorAutoDiff(jacobs_automatic);
    factor_test.TestLoamFeatureFactorAnalytic(jacobs_analytic);
    factor_test.CheckJacobian("LoamFeatureFactor " + descri, jacobs_automatic,
                              jacobs_analytic);

    if (1 == i) {
      factor_test.pc_corr_.t_point = 0.55;
      factor_test.pc_corr_.t_map = 0.33;
    }
    jacobs_automatic.clear();
    jacobs_analytic.clear();
    factor_test.TestLoamFeatureFactorAutoDiff(jacobs_automatic);
    factor_test.TestRalativeLoamFeatureFactorAnalytic(jacobs_analytic);
    factor_test.CheckJacobian("RalativeLoamFeatureFactor " + descri,
                              jacobs_automatic, jacobs_analytic);
  }

  for (int i = 0; i < 2; ++i) {
    Eigen::aligned_map<double*, Eigen::MatrixXd> jacobs_automatic;
    Eigen::aligned_map<double*, Eigen::MatrixXd> jacobs_analytic;

    std::string descri;
    if (0 == i) {
      descri = "LoamFeatureOptMapPoseFactor [Plane]";
      factor_test.pc_corr_.geo_type = Plane;
    } else {
      descri = "LoamFeatureOptMapPoseFactor [Line]";
      factor_test.pc_corr_.geo_type = Line;
    }
    factor_test.TestLoamFeatureOptMapPoseFactorAutoDiff(jacobs_automatic);
    factor_test.TestLoamFeatureOptMapPoseFactorAnalytic(jacobs_analytic);
    factor_test.CheckJacobian(descri, jacobs_automatic, jacobs_analytic);
  }

  {
    Eigen::aligned_map<double*, Eigen::MatrixXd> jacobs_automatic;
    Eigen::aligned_map<double*, Eigen::MatrixXd> jacobs_analytic;

    factor_test.TestIMUPoseFactorAutoDiff(jacobs_automatic);
    factor_test.TestIMUPoseFactorAnalytic(jacobs_analytic);
    factor_test.CheckJacobian("IMU Pose factor", jacobs_automatic,
                              jacobs_analytic);
  }

  {
    Eigen::aligned_map<double*, Eigen::MatrixXd> jacobs_automatic;
    Eigen::aligned_map<double*, Eigen::MatrixXd> jacobs_analytic;

    factor_test.TestIMUFactorAutoDiff(jacobs_automatic);
    factor_test.TestIMUFactorAnalytic(jacobs_analytic);
    factor_test.CheckJacobian("IMU Data factor", jacobs_automatic,
                              jacobs_analytic);
  }

  return 0;
}
