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

#include <ceres/ceres.h>
#include <spline/spline_segment.h>
#include <utils/parameter_struct.h>

#include <estimator/factor/analytic_diff/split_spline_view.h>

namespace clic {
namespace analytic_derivative {

class ImageFeatureFactor : public ceres::CostFunction,
                           So3SplineView,
                           RdSplineView {
 public:
  using SO3View = So3SplineView;
  using R3View = RdSplineView;

  using Vec3d = Eigen::Matrix<double, 3, 1>;
  using Mat3d = Eigen::Matrix<double, 3, 3>;
  using SO3d = Sophus::SO3<double>;

  ImageFeatureFactor(const int64_t t_i_ns, const Eigen::Vector3d& p_i,
                     const int64_t t_j_ns, const Eigen::Vector3d& p_j,
                     const SplineMeta<SplineOrder>& spline_meta)
      : t_i_ns_(t_i_ns),
        p_i_(p_i),
        t_j_ns_(t_j_ns),
        p_j_(p_j),
        spline_meta_(spline_meta) {
    assert(init_flag && "ImageFeatureFactor not init param");

    /// [误差维度 2]
    set_num_residuals(2);

    /// [参数维度 2N]: 旋转控制点 N 个 + 平移控制点 N 个
    size_t kont_num = spline_meta.NumParameters();
    for (size_t i = 0; i < kont_num; ++i) {
      mutable_parameter_block_sizes()->push_back(4);
    }
    for (size_t i = 0; i < kont_num; ++i) {
      mutable_parameter_block_sizes()->push_back(3);
    }
    mutable_parameter_block_sizes()->push_back(1);  // inverse depth
    mutable_parameter_block_sizes()->push_back(1);  // time offset
  }

  virtual bool Evaluate(double const* const* parameters, double* residuals,
                        double** jacobians) const {
    typename SO3View::JacobianStruct J_R[2];
    typename R3View::JacobianStruct J_p[2];

    size_t Knot_offset = 2 * spline_meta_.NumParameters();
    double d_inv = parameters[Knot_offset][0];
    double time_offset_in_ns = parameters[Knot_offset + 1][0];
    int64_t ti_corrected_ns = t_i_ns_ + (int64_t)time_offset_in_ns;
    int64_t tj_corrected_ns = t_j_ns_ + (int64_t)time_offset_in_ns;

    size_t kont_num = spline_meta_.NumParameters();

    size_t R_offset[2] = {0, 0};
    size_t P_offset[2] = {0, 0};
    size_t seg_idx[2] = {0, 0};
    {
      double u;
      spline_meta_.ComputeSplineIndex(ti_corrected_ns, R_offset[0], u);
      spline_meta_.ComputeSplineIndex(tj_corrected_ns, R_offset[1], u);

      // 默认最多只有两个 segments
      size_t segment0_knot_num = spline_meta_.segments.at(0).NumParameters();
      for (int i = 0; i < 2; ++i) {
        if (R_offset[i] >= segment0_knot_num) {
          seg_idx[i] = 1;
          R_offset[i] = segment0_knot_num;
        } else {
          R_offset[i] = 0;
        }
        P_offset[i] = R_offset[i] + kont_num;
      }
    }

    Vec3d x_ci = p_i_ / d_inv;
    Vec3d p_Ii = S_CtoI * x_ci + p_CinI;

    SO3d S_IitoG;
    Vec3d p_IiinG = Vec3d::Zero();
    if (jacobians) {
      // rhs = p_Ii
      S_IitoG = SO3View::EvaluateRp(ti_corrected_ns,
                                    spline_meta_.segments.at(seg_idx[0]),
                                    parameters + R_offset[0], &J_R[0]);
      p_IiinG = R3View::evaluate(ti_corrected_ns,
                                 spline_meta_.segments.at(seg_idx[0]),
                                 parameters + P_offset[0], &J_p[0]);
    } else {
      S_IitoG = SO3View::EvaluateRp(ti_corrected_ns,
                                    spline_meta_.segments.at(seg_idx[0]),
                                    parameters + R_offset[0], nullptr);
      p_IiinG = R3View::evaluate(ti_corrected_ns,
                                 spline_meta_.segments.at(seg_idx[0]),
                                 parameters + P_offset[0], nullptr);
    }
    Vec3d p_G = S_IitoG * p_Ii + p_IiinG;
    SO3d S_GtoIj;
    Vec3d p_IjinG = Vec3d::Zero();
    if (jacobians) {
      // rhs = p_G - p_IjinG
      S_GtoIj = SO3View::EvaluateRTp(tj_corrected_ns,
                                     spline_meta_.segments.at(seg_idx[1]),
                                     parameters + R_offset[1], &J_R[1]);
      p_IjinG = R3View::evaluate(tj_corrected_ns,
                                 spline_meta_.segments.at(seg_idx[1]),
                                 parameters + P_offset[1], &J_p[1]);
    } else {
      S_GtoIj = SO3View::EvaluateRTp(tj_corrected_ns,
                                     spline_meta_.segments.at(seg_idx[1]),
                                     parameters + R_offset[1], nullptr);
      p_IjinG = R3View::evaluate(tj_corrected_ns,
                                 spline_meta_.segments.at(seg_idx[1]),
                                 parameters + P_offset[1], nullptr);
    }

    Vec3d gyro_i, gyro_j;
    Vec3d vel_i, vel_j;
    if (jacobians && jacobians[Knot_offset + 1]) {
      gyro_i = SO3View::VelocityBody(ti_corrected_ns,
                                     spline_meta_.segments.at(seg_idx[0]),
                                     parameters + R_offset[0]);
      vel_i = R3View::velocity(ti_corrected_ns,
                               spline_meta_.segments.at(seg_idx[0]),
                               parameters + P_offset[0]);

      gyro_j = SO3View::VelocityBody(tj_corrected_ns,
                                     spline_meta_.segments.at(seg_idx[1]),
                                     parameters + R_offset[1]);
      vel_j = R3View::velocity(tj_corrected_ns,
                               spline_meta_.segments.at(seg_idx[1]),
                               parameters + P_offset[1]);
    }

    SO3d S_ItoC = S_CtoI.inverse();
    SO3d S_GtoCj = S_ItoC * S_GtoIj;
    Vec3d x_j = S_GtoCj * (p_G - p_IjinG) - S_ItoC * p_CinI;
    // Vec3d p_M =
    //     S_CtoI.inverse() * ((S_GtoIj * (p_G - p_IjinG)) - p_CinI);

    Eigen::Map<Eigen::Vector2d> residual(residuals);
    double depth_j_inv = 1.0 / x_j.z();
    residual = (x_j * depth_j_inv).head<2>() - p_j_.head<2>();

    if (jacobians) {
      for (size_t i = 0; i < kont_num; ++i) {
        if (jacobians[i]) {
          Eigen::Map<Eigen::Matrix<double, 2, 4, Eigen::RowMajor>> jac_kont_R(
              jacobians[i]);
          jac_kont_R.setZero();
        }
        if (jacobians[i + kont_num]) {
          Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> jac_kont_p(
              jacobians[i + kont_num]);
          jac_kont_p.setZero();
        }
      }
    }

    if (jacobians) {
      Eigen::Matrix<double, 2, 3> J_v;
      J_v.block<2, 2>(0, 0) = depth_j_inv * Eigen::Matrix2d::Identity();
      J_v.block<2, 1>(0, 2) = -depth_j_inv * depth_j_inv * x_j.head<2>();

      Eigen::Matrix<double, 2, 3> jac_lhs_R[2];
      Eigen::Matrix<double, 2, 3> jac_lhs_P[2];

      // t_i (frame M is coincide with frame Cj)
      jac_lhs_R[0] = -J_v * (S_GtoCj * S_IitoG).matrix() * SO3::hat(p_Ii);
      jac_lhs_P[0] = J_v * S_GtoCj.matrix();

      // t_j
      jac_lhs_R[1] = J_v * S_GtoCj.matrix() * SO3::hat(p_G - p_IjinG);
      jac_lhs_P[1] = -J_v * S_GtoCj.matrix();

      ///[step1] jacobians of control points
      for (int seg = 0; seg < 2; ++seg) {
        /// Rotation control point
        size_t pre_idx_R = R_offset[seg] + J_R[seg].start_idx;
        for (size_t i = 0; i < SplineOrder; i++) {
          size_t idx = pre_idx_R + i;
          if (jacobians[idx]) {
            Eigen::Map<Eigen::Matrix<double, 2, 4, Eigen::RowMajor>> jac_kont_R(
                jacobians[idx]);
            Eigen::Matrix<double, 2, 3, Eigen::RowMajor> J_temp;
            /// 2*3 3*3
            J_temp = jac_lhs_R[seg] * J_R[seg].d_val_d_knot[i];
            J_temp = (sqrt_info * J_temp).eval();

            jac_kont_R.block<2, 3>(0, 0) += J_temp;
          }
        }

        /// position control point
        size_t pre_idx_P = P_offset[seg] + J_p[seg].start_idx;
        for (size_t i = 0; i < SplineOrder; i++) {
          size_t idx = pre_idx_P + i;
          if (jacobians[idx]) {
            Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> jac_kont_p(
                jacobians[idx]);

            Eigen::Matrix<double, 2, 3, Eigen::RowMajor> J_temp;
            /// 1*1 2*3
            J_temp = J_p[seg].d_val_d_knot[i] * jac_lhs_P[seg];
            J_temp = (sqrt_info * J_temp).eval();

            jac_kont_p += J_temp;
          }
        }
      }

      ///[step2] jacobians of inverse depth
      if (jacobians[Knot_offset]) {
        Eigen::Map<Eigen::Matrix<double, 2, 1>> jac_depth_inv(
            jacobians[Knot_offset]);
        jac_depth_inv.setZero();

        Vec3d J_Xm_d = -(S_GtoCj * S_IitoG * S_CtoI).matrix() * x_ci / d_inv;
        /// 2*3 3*1
        jac_depth_inv.block<2, 1>(0, 0) = J_v * J_Xm_d;
        jac_depth_inv = (sqrt_info * jac_depth_inv).eval();
      }

      // [step3] time offset
      if (jacobians[Knot_offset + 1]) {
        Eigen::Map<Eigen::Matrix<double, 2, 1>> jac_t_offset(
            jacobians[Knot_offset + 1]);
        jac_t_offset.setZero();

        Mat3d Ri_dot = S_IitoG.matrix() * SO3d::hat(gyro_i);
        Mat3d Rj_dot = S_GtoIj.inverse().matrix() * SO3d::hat(gyro_j);
        // t_j
        Vec3d J_tj = S_ItoC.matrix() * Rj_dot.transpose() * (p_G - p_IjinG) -
                     S_GtoCj * vel_j;
        // t_i
        Vec3d J_ti = S_GtoCj.matrix() * (Ri_dot * p_Ii + vel_i);
        jac_t_offset = 1e-9 * sqrt_info * J_v * (J_ti + J_tj);
      }
    }

    residual = (sqrt_info * residual).eval();
    return true;
  }

  static void SetParam(SO3d _S_CtoI, Vec3d _p_CinI) {
    init_flag = true;
    S_CtoI = _S_CtoI;
    p_CinI = _p_CinI;
  }

  // double focal_length = 450.;
  static inline Eigen::Matrix2d sqrt_info =
      450. / 1.5 * Eigen::Matrix2d::Identity();

 private:
  static inline bool init_flag = false;

  static inline SO3d S_CtoI;
  static inline Vec3d p_CinI;

  int64_t t_i_ns_;
  Eigen::Vector3d p_i_;
  int64_t t_j_ns_;
  Eigen::Vector3d p_j_;

  SplineMeta<SplineOrder> spline_meta_;
};

class Image3D2DFactor : public ceres::CostFunction,
                        So3SplineView,
                        RdSplineView {
 public:
  using SO3View = So3SplineView;
  using R3View = RdSplineView;

  using Vec3d = Eigen::Matrix<double, 3, 1>;
  using Mat3d = Eigen::Matrix<double, 3, 3>;
  using SO3d = Sophus::SO3<double>;

  Image3D2DFactor(const int64_t t_j_ns, const Eigen::Vector3d& p_j,
                  const SplineMeta<SplineOrder>& spline_meta)
      : t_j_ns_(t_j_ns), p_j_(p_j), spline_meta_(spline_meta) {
    assert(init_flag && "Image3D2DFactor not init param");

    set_num_residuals(2);

    size_t kont_num = spline_meta.NumParameters();
    for (size_t i = 0; i < kont_num; ++i) {
      mutable_parameter_block_sizes()->push_back(4);
    }
    for (size_t i = 0; i < kont_num; ++i) {
      mutable_parameter_block_sizes()->push_back(3);
    }
    mutable_parameter_block_sizes()->push_back(3);  // p_inG
    mutable_parameter_block_sizes()->push_back(1);  // time offset
  }

  virtual bool Evaluate(double const* const* parameters, double* residuals,
                        double** jacobians) const {
    typename SO3View::JacobianStruct J_R;
    typename R3View::JacobianStruct J_p;

    size_t Knot_offset = 2 * spline_meta_.NumParameters();
    Eigen::Map<const Vec3d> p_G(parameters[Knot_offset]);
    double time_offset_in_ns = parameters[Knot_offset + 1][0];

    int64_t tj_corrected_ns = t_j_ns_ + (int64_t)time_offset_in_ns;

    size_t kont_num = spline_meta_.NumParameters();

    ///  t_j 
    SO3d S_GtoIj;
    Vec3d p_IjinG = Vec3d::Zero();
    if (jacobians) {
      // rhs = p_G - p_IjinG
      S_GtoIj = SO3View::EvaluateRTp(
          tj_corrected_ns, spline_meta_.segments.at(0), parameters, &J_R);
      p_IjinG = R3View::evaluate(tj_corrected_ns, spline_meta_.segments.at(0),
                                 parameters + kont_num, &J_p);
    } else {
      S_GtoIj = SO3View::EvaluateRTp(
          tj_corrected_ns, spline_meta_.segments.at(0), parameters, nullptr);
      p_IjinG = R3View::evaluate(tj_corrected_ns, spline_meta_.segments.at(0),
                                 parameters + kont_num, nullptr);
    }

    Vec3d gyro_j, vel_j;
    if (jacobians && jacobians[Knot_offset + 1]) {
      gyro_j = SO3View::VelocityBody(tj_corrected_ns,
                                     spline_meta_.segments.at(0), parameters);
      vel_j = R3View::velocity(tj_corrected_ns, spline_meta_.segments.at(0),
                               parameters + kont_num);
    }

    SO3d S_ItoC = S_CtoI.inverse();
    SO3d S_GtoCj = S_ItoC * S_GtoIj;
    Vec3d x_j = S_GtoCj * (p_G - p_IjinG) - S_ItoC * p_CinI;

    Eigen::Map<Eigen::Vector2d> residual(residuals);
    double depth_j_inv = 1.0 / x_j.z();
    residual = (x_j * depth_j_inv).head<2>() - p_j_.head<2>();

    if (jacobians) {
      for (size_t i = 0; i < kont_num; ++i) {
        if (jacobians[i]) {
          Eigen::Map<Eigen::Matrix<double, 2, 4, Eigen::RowMajor>> jac_kont_R(
              jacobians[i]);
          jac_kont_R.setZero();
        }
        if (jacobians[i + kont_num]) {
          Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> jac_kont_p(
              jacobians[i + kont_num]);
          jac_kont_p.setZero();
        }
      }
    }

    if (jacobians) {
      Eigen::Matrix<double, 2, 3> J_v;
      J_v.block<2, 2>(0, 0) = depth_j_inv * Eigen::Matrix2d::Identity();
      J_v.block<2, 1>(0, 2) = -depth_j_inv * depth_j_inv * x_j.head<2>();

      Eigen::Matrix<double, 2, 3> jac_lhs_R;
      Eigen::Matrix<double, 2, 3> jac_lhs_P;

      //  t_j 
      jac_lhs_R = J_v * S_GtoCj.matrix() * SO3::hat(p_G - p_IjinG);
      jac_lhs_P = -J_v * S_GtoCj.matrix();

      /// Rotation control point
      for (size_t i = 0; i < SplineOrder; i++) {
        size_t idx = J_R.start_idx + i;
        if (jacobians[idx]) {
          Eigen::Map<Eigen::Matrix<double, 2, 4, Eigen::RowMajor>> jac_kont_R(
              jacobians[idx]);
          Eigen::Matrix<double, 2, 3, Eigen::RowMajor> J_temp;
          /// 2*3 3*3
          J_temp = jac_lhs_R * J_R.d_val_d_knot[i];
          J_temp = (sqrt_info * J_temp).eval();

          jac_kont_R.block<2, 3>(0, 0) += J_temp;
        }
      }

      /// position control point
      for (size_t i = 0; i < SplineOrder; i++) {
        size_t idx = J_p.start_idx + kont_num + i;
        if (jacobians[idx]) {
          Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> jac_kont_p(
              jacobians[idx]);

          Eigen::Matrix<double, 2, 3, Eigen::RowMajor> J_temp;
          /// 1*1 2*3
          J_temp = J_p.d_val_d_knot[i] * jac_lhs_P;
          J_temp = (sqrt_info * J_temp).eval();

          jac_kont_p += J_temp;
        }
      }

      /// jacobian of p_G
      if (jacobians[Knot_offset]) {
        Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> jac_p_G(
            jacobians[Knot_offset]);
        jac_p_G.setZero();

        /// 2*3 3*3
        jac_p_G = J_v * S_GtoCj.matrix();
        jac_p_G = (sqrt_info * jac_p_G).eval();
      }

      // jacobian of time offset
      if (jacobians[Knot_offset + 1]) {
        Eigen::Map<Eigen::Matrix<double, 2, 1>> jac_t_offset(
            jacobians[Knot_offset + 1]);
        jac_t_offset.setZero();

        Mat3d Rj_dot = S_GtoIj.inverse().matrix() * SO3d::hat(gyro_j);
        // t_j
        Vec3d J_tj = S_ItoC.matrix() * Rj_dot.transpose() * (p_G - p_IjinG) -
                     S_GtoCj * vel_j;
        jac_t_offset = 1e-9 * sqrt_info * J_v * J_tj;
      }
    }

    residual = (sqrt_info * residual).eval();
    return true;
  }

  static void SetParam(SO3d _S_CtoI, Vec3d _p_CinI) {
    init_flag = true;
    S_CtoI = _S_CtoI;
    p_CinI = _p_CinI;
  }

  // double focal_length = 450.;
  static inline Eigen::Matrix2d sqrt_info =
      450. / 1.5 * Eigen::Matrix2d::Identity();

 private:
  static inline bool init_flag = false;

  static inline SO3d S_CtoI;
  static inline Vec3d p_CinI;

  int64_t t_j_ns_;
  Eigen::Vector3d p_j_;

  SplineMeta<SplineOrder> spline_meta_;
};

class ImageFeatureOnePoseFactor : public ceres::CostFunction,
                                  So3SplineView,
                                  RdSplineView {
 public:
  using SO3View = So3SplineView;
  using R3View = RdSplineView;

  using Vec3d = Eigen::Matrix<double, 3, 1>;
  using SO3d = Sophus::SO3<double>;

  ImageFeatureOnePoseFactor(const Eigen::Vector3d& p_i, const SO3d& S_IitoG,
                            const Eigen::Vector3d& p_IiinG,
                            const int64_t t_j_ns, const Eigen::Vector3d& p_j,
                            const SplineMeta<SplineOrder>& spline_meta)
      : p_i_(p_i),
        S_IitoG_(S_IitoG),
        p_IiinG_(p_IiinG),
        t_j_ns_(t_j_ns),
        p_j_(p_j),
        spline_meta_(spline_meta) {
    assert(init_flag && "ImageFeatureOnePoseFactor not init param");

    set_num_residuals(2);

    size_t kont_num = spline_meta.NumParameters();
    for (size_t i = 0; i < kont_num; ++i) {
      mutable_parameter_block_sizes()->push_back(4);
    }
    for (size_t i = 0; i < kont_num; ++i) {
      mutable_parameter_block_sizes()->push_back(3);
    }
    mutable_parameter_block_sizes()->push_back(1);  // inverse depth
  }

  virtual bool Evaluate(double const* const* parameters, double* residuals,
                        double** jacobians) const {
    typename SO3View::JacobianStruct J_R;
    typename R3View::JacobianStruct J_p;

    size_t Knot_offset = 2 * spline_meta_.NumParameters();
    double d_inv = parameters[Knot_offset][0];

    size_t kont_num = spline_meta_.NumParameters();

    size_t R_offset, P_offset;

    double u;
    spline_meta_.ComputeSplineIndex(t_j_ns_, R_offset, u);
    P_offset = R_offset + kont_num;

    Vec3d x_ci = p_i_ / d_inv;
    ///  t_i 
    Vec3d p_Ii = S_CtoI * x_ci + p_CinI;

    /// t_j 
    Vec3d p_G = S_IitoG_ * p_Ii + p_IiinG_;
    SO3d S_GtoIj;
    Vec3d p_IjinG = Vec3d::Zero();
    if (jacobians) {
      // rhs = p_G - p_IjinG
      S_GtoIj = SO3View::EvaluateRTp(t_j_ns_, spline_meta_.segments.at(0),
                                     parameters + R_offset, &J_R);
      p_IjinG = R3View::evaluate(t_j_ns_, spline_meta_.segments.at(0),
                                 parameters + P_offset, &J_p);
    } else {
      S_GtoIj = SO3View::EvaluateRTp(t_j_ns_, spline_meta_.segments.at(0),
                                     parameters + R_offset, nullptr);
      p_IjinG = R3View::evaluate(t_j_ns_, spline_meta_.segments.at(0),
                                 parameters + P_offset, nullptr);
    }
    SO3d S_ItoC = S_CtoI.inverse();
    SO3d S_GtoCj = S_ItoC * S_GtoIj;
    Vec3d x_j = S_GtoCj * (p_G - p_IjinG) - S_ItoC * p_CinI;

    Eigen::Map<Eigen::Vector2d> residual(residuals);
    double depth_j_inv = 1.0 / x_j.z();
    residual = (x_j * depth_j_inv).head<2>() - p_j_.head<2>();

    if (jacobians) {
      for (size_t i = 0; i < kont_num; ++i) {
        if (jacobians[i]) {
          Eigen::Map<Eigen::Matrix<double, 2, 4, Eigen::RowMajor>> jac_kont_R(
              jacobians[i]);
          jac_kont_R.setZero();
        }
        if (jacobians[i + kont_num]) {
          Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> jac_kont_p(
              jacobians[i + kont_num]);
          jac_kont_p.setZero();
        }
      }
    }

    if (jacobians) {
      Eigen::Matrix<double, 2, 3> J_v;
      J_v.block<2, 2>(0, 0) = depth_j_inv * Eigen::Matrix2d::Identity();
      J_v.block<2, 1>(0, 2) = -depth_j_inv * depth_j_inv * x_j.head<2>();

      Eigen::Matrix<double, 2, 3> jac_lhs_R, jac_lhs_P;
      //  t_j 
      jac_lhs_R = J_v * S_GtoCj.matrix() * SO3::hat(p_G - p_IjinG);
      jac_lhs_P = -J_v * S_GtoCj.matrix();

      ///[step1] jacobians of control points

      /// Rotation control point
      size_t pre_idx_R = R_offset + J_R.start_idx;
      for (size_t i = 0; i < SplineOrder; i++) {
        size_t idx = pre_idx_R + i;
        if (jacobians[idx]) {
          Eigen::Map<Eigen::Matrix<double, 2, 4, Eigen::RowMajor>> jac_kont_R(
              jacobians[idx]);
          Eigen::Matrix<double, 2, 3, Eigen::RowMajor> J_temp;
          /// 2*3 3*3
          J_temp = jac_lhs_R * J_R.d_val_d_knot[i];
          J_temp = (sqrt_info * J_temp).eval();

          jac_kont_R.block<2, 3>(0, 0) += J_temp;
        }
      }

      /// position control point
      size_t pre_idx_P = P_offset + J_p.start_idx;
      for (size_t i = 0; i < SplineOrder; i++) {
        size_t idx = pre_idx_P + i;
        if (jacobians[idx]) {
          Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> jac_kont_p(
              jacobians[idx]);

          Eigen::Matrix<double, 2, 3, Eigen::RowMajor> J_temp;
          /// 1*1 2*3
          J_temp = J_p.d_val_d_knot[i] * jac_lhs_P;
          J_temp = (sqrt_info * J_temp).eval();

          jac_kont_p += J_temp;
        }
      }

      ///[step2] jacobians of inverse depth
      if (jacobians[Knot_offset]) {
        Eigen::Map<Eigen::Matrix<double, 2, 1>> jac_depth_inv(
            jacobians[Knot_offset]);
        jac_depth_inv.setZero();

        Vec3d J_Xm_d = -(S_GtoCj * S_IitoG_ * S_CtoI).matrix() * x_ci / d_inv;
        /// 2*3 3*1
        jac_depth_inv.block<2, 1>(0, 0) = J_v * J_Xm_d;
        jac_depth_inv = (sqrt_info * jac_depth_inv).eval();
      }
    }

    residual = (sqrt_info * residual).eval();
    return true;
  }

  static void SetParam(SO3d _S_CtoI, Vec3d _p_CinI) {
    init_flag = true;
    S_CtoI = _S_CtoI;
    p_CinI = _p_CinI;
  }

  // double focal_length = 450.;
  static inline Eigen::Matrix2d sqrt_info =
      450. / 1.5 * Eigen::Matrix2d::Identity();

 private:
  static inline bool init_flag = false;

  static inline SO3d S_CtoI;
  static inline Vec3d p_CinI;

  Eigen::Vector3d p_i_;
  SO3d S_IitoG_;
  Eigen::Vector3d p_IiinG_;

  int64_t t_j_ns_;
  Eigen::Vector3d p_j_;

  SplineMeta<SplineOrder> spline_meta_;
};

class ImageDepthFactor : public ceres::SizedCostFunction<2, 1> {
 public:
  using Vec3d = Eigen::Matrix<double, 3, 1>;

  ImageDepthFactor(const Eigen::Vector3d& p_i, const Eigen::Vector3d& p_j,
                   const SO3d& S_CitoCj, const Eigen::Vector3d& p_CiinCj)
      : p_i_(p_i), p_j_(p_j), S_CitoCj_(S_CitoCj), p_CiinCj_(p_CiinCj) {}

  virtual bool Evaluate(double const* const* parameters, double* residuals,
                        double** jacobians) const {
    double d_inv = parameters[0][0];

    Vec3d x_ci = p_i_ / d_inv;
    Vec3d x_j = S_CitoCj_ * x_ci + p_CiinCj_;
    double depth_j_inv = 1.0 / x_j.z();

    Eigen::Map<Eigen::Vector2d> residual(residuals);
    residual = (x_j * depth_j_inv).head<2>() - p_j_.head<2>();

    if (jacobians) {
      Eigen::Matrix<double, 2, 3> J_v;
      J_v.block<2, 2>(0, 0) = depth_j_inv * Eigen::Matrix2d::Identity();
      J_v.block<2, 1>(0, 2) = -depth_j_inv * depth_j_inv * x_j.head<2>();

      /// jacobians of inverse depth
      if (jacobians[0]) {
        Eigen::Map<Eigen::Matrix<double, 2, 1>> jac_depth_inv(jacobians[0]);
        jac_depth_inv.setZero();

        Vec3d J_Xm_d = -S_CitoCj_.matrix() * x_ci / d_inv;
        /// 2*3 3*1
        jac_depth_inv.block<2, 1>(0, 0) = J_v * J_Xm_d;
        jac_depth_inv = (sqrt_info * jac_depth_inv).eval();
      }
    }

    residual = (sqrt_info * residual).eval();
    return true;
  }

  // double focal_length = 450.;
  static inline Eigen::Matrix2d sqrt_info =
      450. / 1.5 * Eigen::Matrix2d::Identity();

 private:
  Eigen::Vector3d p_i_;
  Eigen::Vector3d p_j_;
  SO3d S_CitoCj_;
  Eigen::Vector3d p_CiinCj_;
};

class EpipolarFactor : public ceres::CostFunction, So3SplineView, RdSplineView {
 public:
  using SO3View = So3SplineView;
  using R3View = RdSplineView;

  using Vec3d = Eigen::Matrix<double, 3, 1>;
  using Mat3d = Eigen::Matrix<double, 3, 3>;
  using SO3d = Sophus::SO3<double>;

  EpipolarFactor(const int64_t t_i_ns, const Eigen::Vector3d& x_i,
                 const Eigen::Vector3d& x_k, const SO3d& S_GtoCk,
                 const Eigen::Vector3d& p_CkinG,
                 const SplineMeta<SplineOrder>& spline_meta, double weight)
      : t_i_ns_(t_i_ns),
        x_i_(x_i),
        x_k_(x_k),
        S_GtoCk_(S_GtoCk),
        p_CkinG_(p_CkinG),
        spline_meta_(spline_meta),
        weight_(weight) {
    assert(init_flag && "EpipolarFactor not init param");
    set_num_residuals(1);

    size_t kont_num = spline_meta.NumParameters();
    for (size_t i = 0; i < kont_num; ++i) {
      mutable_parameter_block_sizes()->push_back(4);
    }
    for (size_t i = 0; i < kont_num; ++i) {
      mutable_parameter_block_sizes()->push_back(3);
    }
  }

  virtual bool Evaluate(double const* const* parameters, double* residuals,
                        double** jacobians) const {
    typename SO3View::JacobianStruct J_R;
    typename R3View::JacobianStruct J_p;

    size_t kont_num = spline_meta_.NumParameters();

    SO3d S_IitoG;
    Vec3d p_IiinG = Vec3d::Zero();
    if (jacobians) {
      S_IitoG = SO3View::EvaluateRp(t_i_ns_, spline_meta_.segments[0],
                                    parameters, &J_R);
      p_IiinG = R3View::evaluate(t_i_ns_, spline_meta_.segments[0],
                                 parameters + kont_num, &J_p);
    } else {
      S_IitoG = SO3View::EvaluateRp(t_i_ns_, spline_meta_.segments[0],
                                    parameters, nullptr);
      p_IiinG = R3View::evaluate(t_i_ns_, spline_meta_.segments[0],
                                 parameters + kont_num, nullptr);
    }

    Vec3d Rxi = S_GtoCk_ * S_IitoG * S_CtoI * x_i_;
    Eigen::Matrix3d t_hat =
        SO3::hat(S_GtoCk_ * (S_IitoG * p_CinI + p_IiinG - p_CkinG_));
    residuals[0] = x_k_.transpose() * t_hat * Rxi;

    residuals[0] *= weight_;

    if (jacobians) {
      for (size_t i = 0; i < kont_num; ++i) {
        if (jacobians[i]) {
          Eigen::Map<Eigen::Matrix<double, 1, 4, Eigen::RowMajor>> jac_kont_R(
              jacobians[i]);
          jac_kont_R.setZero();
        }
        if (jacobians[i + kont_num]) {
          Eigen::Map<Eigen::Matrix<double, 1, 3, Eigen::RowMajor>> jac_kont_p(
              jacobians[i + kont_num]);
          jac_kont_p.setZero();
        }
      }

      Vec3d jac_lhs = x_k_.transpose() * SO3::hat(Rxi) * S_GtoCk_.matrix();
      Vec3d jac_lhs_P = -jac_lhs;
      Vec3d jac_lhs_R = -x_k_.transpose() * t_hat *
                        (S_GtoCk_ * S_IitoG).matrix() * SO3::hat(S_CtoI * x_i_);
      jac_lhs_R += jac_lhs.transpose() * S_IitoG.matrix() * SO3::hat(p_CinI);

      /// Rotation control point
      for (size_t i = 0; i < kont_num; i++) {
        size_t idx = i;
        if (jacobians[idx]) {
          Eigen::Map<Eigen::Matrix<double, 1, 4, Eigen::RowMajor>> jac_kont_R(
              jacobians[idx]);
          jac_kont_R.setZero();

          /// 1*3 3*3
          jac_kont_R.block<1, 3>(0, 0) =
              jac_lhs_R.transpose() * J_R.d_val_d_knot[i];
          jac_kont_R = (weight_ * jac_kont_R).eval();
        }
      }

      /// position control point
      for (size_t i = 0; i < kont_num; i++) {
        size_t idx = kont_num + i;
        if (jacobians[idx]) {
          Eigen::Map<Eigen::Matrix<double, 1, 3, Eigen::RowMajor>> jac_kont_p(
              jacobians[idx]);
          jac_kont_p.setZero();

          /// 1*1 1*3
          jac_kont_p = J_p.d_val_d_knot[i] * jac_lhs_P;
          jac_kont_p = (weight_ * jac_kont_p).eval();
        }
      }
    }

    return true;
  }

  static void SetParam(SO3d _S_CtoI, Vec3d _p_CinI) {
    init_flag = true;
    S_CtoI = _S_CtoI;
    p_CinI = _p_CinI;
  }

 private:
  static inline bool init_flag = false;
  static inline SO3d S_CtoI;
  static inline Vec3d p_CinI;

  int64_t t_i_ns_;
  Eigen::Vector3d x_i_;

  Eigen::Vector3d x_k_;
  SO3d S_GtoCk_;
  Eigen::Vector3d p_CkinG_;

  SplineMeta<SplineOrder> spline_meta_;
  double weight_;
};

}  // namespace analytic_derivative

}  // namespace clic
