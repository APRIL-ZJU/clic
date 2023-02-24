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
#include <spline/ceres_spline_helper.h>
#include <spline/ceres_spline_helper_jet.h>
#include <spline/spline_segment.h>
#include <utils/parameter_struct.h>

#include <sophus_lib/so3.hpp>

#include "visual_odometry/integration_base.h"

namespace clic {

namespace gravity_factor {

// Eigen::Matrix<ceres::Jet<double, 1>, 6, 1> gravity_jet;
template <typename T>
T gravity_jet;  //= gravity_double.template cast<T>();

template <typename T>
Eigen::Matrix<T, 3, 2> TangentBasis(
    Eigen::Map<const Eigen::Matrix<T, 3, 1>>& g0) {
  Eigen::Matrix<T, 3, 1> tmp(T(0), T(0), T(-1));

  Eigen::Matrix<T, 3, 1> a = g0.normalized();
  if (a == tmp) tmp << T(-1), T(0), T(0);

  Eigen::Matrix<T, 3, 1> b = (tmp - a * (a.transpose() * tmp)).normalized();
  Eigen::Matrix<T, 3, 1> c = a.cross(b);
  Eigen::Matrix<T, 3, 2> bc;
  bc.template block<3, 1>(0, 0) = b;
  bc.template block<3, 1>(0, 1) = c;

  return bc;
}

template <typename T>
Eigen::Matrix<T, 3, 1> refined_gravity(
    Eigen::Map<const Eigen::Matrix<T, 2, 1>>& g_param) {
  T cr = ceres::cos(g_param[0]);
  T sr = ceres::sin(g_param[0]);
  T cp = ceres::cos(g_param[1]);
  T sp = ceres::sin(g_param[1]);
  return Eigen::Matrix<T, 3, 1>(-sp * cr * T(GRAVITY_NORM),
                                sr * T(GRAVITY_NORM),
                                -cr * cp * T(GRAVITY_NORM));
}

inline Eigen::Vector2d recover_gravity_param(Eigen::Vector3d& gravity) {
  Eigen::Vector2d g_param;
  double gravity_norm =
      -std::sqrt(gravity(0) * gravity(0) + gravity(1) * gravity(1) +
                 gravity(2) * gravity(2));
  double sr = gravity(1) / gravity_norm;
  g_param(0) = asin(sr);
  double sp = -gravity(0) / gravity_norm / cos(g_param(0));
  g_param(1) = asin(sp);
  return g_param;
}
}  // namespace gravity_factor

namespace auto_diff {

class GyroFactor {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  GyroFactor(int64_t time_ns, const IMUData& imu_data,
             const SplineMeta<SplineOrder>& spline_meta, double gyro_weight)
      : time_ns_(time_ns),
        imu_data_(imu_data),
        spline_meta_(spline_meta),
        gyro_weight_(gyro_weight) {
    inv_dt_ = 1e9 * 1.0 / spline_meta_.segments.begin()->dt_ns;
  }

  template <class T>
  bool operator()(T const* const* sKnots, T* sResiduals) const {
    using Tangent = typename Sophus::SO3<T>::Tangent;
    using Vec3T = Eigen::Matrix<T, 3, 1>;

    size_t knot_num = spline_meta_.NumParameters();
    Eigen::Map<const Vec3T> gyro_bias(sKnots[knot_num]);
    T time_offset_in_ns = sKnots[knot_num][0];
    T t_corrected = T(time_ns_) + time_offset_in_ns;

    size_t R_offset;  
    T u;
    spline_meta_.ComputeSplineIndex(t_corrected, R_offset, u);

    Sophus::SO3<T> R_w_i;
    Tangent rot_vel;
    CeresSplineHelperJet<T, SplineOrder>::template evaluate_lie<Sophus::SO3>(
        sKnots + R_offset, u, inv_dt_, &R_w_i, &rot_vel);

    Eigen::Map<Eigen::Matrix<T, 3, 1>> residuals(sResiduals);
    Vec3T r_gyro = rot_vel - imu_data_.gyro.template cast<T>() + gyro_bias;
    residuals.template block<3, 1>(0, 0) = T(gyro_weight_) * r_gyro;

    return true;
  }

  static auto* Create(int64_t time_ns, const IMUData& imu_data,
                      const SplineMeta<SplineOrder>& spline_meta,
                      double gyro_weight) {
    using Functor = GyroFactor;
    return (new ceres::DynamicAutoDiffCostFunction<Functor>(
        new Functor(time_ns, imu_data, spline_meta, gyro_weight)));
  }

 private:
  int64_t time_ns_;
  IMUData imu_data_;
  SplineMeta<SplineOrder> spline_meta_;
  double gyro_weight_;
  double inv_dt_;
};

class GyroAcceWithConstantBiasFactor {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  GyroAcceWithConstantBiasFactor(int64_t time_ns, const IMUData& imu_data,
                                 const SplineMeta<SplineOrder>& spline_meta,
                                 double gyro_weight, double acce_weight)
      : time_ns_(time_ns),
        imu_data_(imu_data),
        spline_meta_(spline_meta),
        gyro_weight_(gyro_weight),
        acce_weight_(acce_weight) {
    inv_dt_ = 1e9 * 1.0 / spline_meta_.segments.begin()->dt_ns;
  }

  template <class T>
  bool operator()(T const* const* sKnots, T* sResiduals) const {
    using Tangent = typename Sophus::SO3<T>::Tangent;
    using Vec3T = Eigen::Matrix<T, 3, 1>;

    size_t Knot_offset = 2 * spline_meta_.NumParameters();
    Eigen::Map<const Vec3T> gyro_bias(sKnots[Knot_offset]);
    Eigen::Map<const Vec3T> acce_bias(sKnots[Knot_offset + 1]);
    Eigen::Map<const Eigen::Matrix<T, 2, 1>> g_refine(sKnots[Knot_offset + 2]);
    T time_offset_in_ns = sKnots[Knot_offset + 3][0];
    T t_corrected = T(time_ns_) + time_offset_in_ns;

    Eigen::Map<Eigen::Matrix<T, 6, 1>> residuals(sResiduals);

    size_t R_offset;
    size_t P_offset;
    T u;
    spline_meta_.ComputeSplineIndex(t_corrected, R_offset, u);
    P_offset = R_offset + spline_meta_.NumParameters();

    Sophus::SO3<T> R_w_i;
    Tangent rot_vel;
    CeresSplineHelperJet<T, SplineOrder>::template evaluate_lie<Sophus::SO3>(
        sKnots + R_offset, u, inv_dt_, &R_w_i, &rot_vel);

    Vec3T accel_w;
    CeresSplineHelperJet<T, SplineOrder>::template evaluate<3, 2>(
        sKnots + P_offset, u, inv_dt_, &accel_w);

#if true

    Vec3T gravity = gravity_factor::refined_gravity(g_refine);

#else

    Eigen::Matrix<T, 3, 2> lxly = gravity_factor::TangentBasis(gravity);
    Vec3T g_opt = (gravity + lxly * g_refine).normalized() * T(9.8);
    gravity_factor::gg = g_opt;
    // gravity_factor::gravity_jet<T> = g_opt;
    // gravity = g_opt;
#endif
    Vec3T r_gyro = rot_vel - imu_data_.gyro.template cast<T>() + gyro_bias;
    Vec3T r_acce = R_w_i.inverse() * (accel_w + gravity) -
                   imu_data_.accel.template cast<T>() + acce_bias;

    residuals.template block<3, 1>(0, 0) = T(gyro_weight_) * r_gyro;
    residuals.template block<3, 1>(3, 0) = T(acce_weight_) * r_acce;

    return true;
  }

  static auto* Create(int64_t time_ns, const IMUData& imu_data,
                      const SplineMeta<SplineOrder>& spline_meta,
                      double gyro_weight, double acce_weight) {
    using Functor = GyroAcceWithConstantBiasFactor;
    return (new ceres::DynamicAutoDiffCostFunction<Functor>(
        new Functor(time_ns, imu_data, spline_meta, gyro_weight, acce_weight)));
  }

 private:
  int64_t time_ns_;
  IMUData imu_data_;
  SplineMeta<SplineOrder> spline_meta_;
  double gyro_weight_;
  double acce_weight_;
  double inv_dt_;
};

class IMUFactor {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  IMUFactor(int64_t time_ns, const IMUData& imu_data,
            const SplineMeta<SplineOrder>& spline_meta, double gyro_weight,
            double acce_weight)
      : time_ns_(time_ns),
        imu_data_(imu_data),
        spline_meta_(spline_meta),
        gyro_weight_(gyro_weight),
        acce_weight_(acce_weight) {
    inv_dt_ = 1e9 * 1.0 / spline_meta_.segments.begin()->dt_ns;
  }

  template <class T>
  bool operator()(T const* const* sKnots, T* sResiduals) const {
    using Tangent = typename Sophus::SO3<T>::Tangent;
    using Vec3T = Eigen::Matrix<T, 3, 1>;

    size_t Knot_offset = 2 * spline_meta_.NumParameters();
    Eigen::Map<const Vec3T> gyro_bias(sKnots[Knot_offset]);
    Eigen::Map<const Vec3T> acce_bias(sKnots[Knot_offset + 1]);
    Eigen::Map<const Vec3T> gravity(sKnots[Knot_offset + 2]);
    T time_offset_in_ns = sKnots[Knot_offset + 3][0];
    T t_corrected = T(time_ns_) + time_offset_in_ns;

    size_t R_offset;
    size_t P_offset;
    T u;
    spline_meta_.ComputeSplineIndex(t_corrected, R_offset, u);
    P_offset = R_offset + spline_meta_.NumParameters();

    Sophus::SO3<T> R_w_i;
    Tangent rot_vel;
    CeresSplineHelperJet<T, SplineOrder>::template evaluate_lie<Sophus::SO3>(
        sKnots + R_offset, u, inv_dt_, &R_w_i, &rot_vel);

    Vec3T accel_w;
    CeresSplineHelperJet<T, SplineOrder>::template evaluate<3, 2>(
        sKnots + P_offset, u, inv_dt_, &accel_w);

    Vec3T r_gyro = rot_vel - imu_data_.gyro.template cast<T>() + gyro_bias;
    Vec3T r_acce = R_w_i.inverse() * (accel_w + gravity) -
                   imu_data_.accel.template cast<T>() + acce_bias;

    Eigen::Map<Eigen::Matrix<T, 6, 1>> residuals(sResiduals);
    residuals.template block<3, 1>(0, 0) = T(gyro_weight_) * r_gyro;
    residuals.template block<3, 1>(3, 0) = T(acce_weight_) * r_acce;

    return true;
  }

  static auto* Create(int64_t time_ns, const IMUData& imu_data,
                      const SplineMeta<SplineOrder>& spline_meta,
                      double gyro_weight, double acce_weight) {
    using Functor = IMUFactor;
    return (new ceres::DynamicAutoDiffCostFunction<Functor>(
        new Functor(time_ns, imu_data, spline_meta, gyro_weight, acce_weight)));
  }

 private:
  int64_t time_ns_;
  IMUData imu_data_;
  SplineMeta<SplineOrder> spline_meta_;
  double gyro_weight_;
  double acce_weight_;
  double inv_dt_;
};

class GyroAcceBiasFactor {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  GyroAcceBiasFactor(int64_t time_ns, const IMUData& imu_data,
                     const IMUBias& imu_bias,
                     const SplineMeta<SplineOrder>& spline_meta,
                     double gyro_weight, double acce_weight, double bias_weight)
      : time_ns_(time_ns),
        imu_data_(imu_data),
        imu_bias_(imu_bias),
        spline_meta_(spline_meta),
        gyro_weight_(gyro_weight),
        acce_weight_(acce_weight),
        bias_weight_(bias_weight) {
    inv_dt_ = 1e9 * 1.0 / spline_meta_.segments.begin()->dt_ns;
  }

  template <class T>
  bool operator()(T const* const* sKnots, T* sResiduals) const {
    using Vec2T = Eigen::Matrix<T, 2, 1>;
    using Vec3T = Eigen::Matrix<T, 3, 1>;
    using Tangent = typename Sophus::SO3<T>::Tangent;

    size_t Knot_offset = 2 * spline_meta_.NumParameters();
    Eigen::Map<const Vec3T> gyro_bias(sKnots[Knot_offset]);
    Eigen::Map<const Vec3T> acce_bias(sKnots[Knot_offset + 1]);
    Eigen::Map<const Vec2T> g_refine(sKnots[Knot_offset + 2]);
    T time_offset_in_ns = sKnots[Knot_offset + 3][0];
    T t_corrected = T(time_ns_) + time_offset_in_ns;

    size_t R_offset;  // should be zero if not estimate time offset
    size_t P_offset;
    T u;
    spline_meta_.ComputeSplineIndex(t_corrected, R_offset, u);
    P_offset = R_offset + spline_meta_.NumParameters();

    Sophus::SO3<T> R_w_i;
    Tangent rot_vel;
    CeresSplineHelperJet<T, SplineOrder>::template evaluate_lie<Sophus::SO3>(
        sKnots + R_offset, u, inv_dt_, &R_w_i, &rot_vel);

    Vec3T accel_w;
    CeresSplineHelperJet<T, SplineOrder>::template evaluate<3, 2>(
        sKnots + P_offset, u, inv_dt_, &accel_w);

    Vec3T gravity = gravity_factor::refined_gravity(g_refine);
    Vec3T gyro_residuals =
        rot_vel - imu_data_.gyro.template cast<T>() + gyro_bias;
    Vec3T acce_residuals = R_w_i.inverse() * (accel_w + gravity) -
                           imu_data_.accel.template cast<T>() + acce_bias;

    Eigen::Map<Eigen::Matrix<T, 12, 1>> residuals(sResiduals);
    residuals.template block<3, 1>(0, 0) = T(gyro_weight_) * gyro_residuals;
    residuals.template block<3, 1>(3, 0) = T(acce_weight_) * acce_residuals;
    residuals.template block<3, 1>(6, 0) =
        T(bias_weight_) * (gyro_bias - imu_bias_.gyro_bias.template cast<T>());
    residuals.template block<3, 1>(9, 0) =
        T(bias_weight_) * (acce_bias - imu_bias_.accel_bias.template cast<T>());

    return true;
  }

  static auto* Create(int64_t time_ns, const IMUData& imu_data,
                      const IMUBias& imu_bias,
                      const SplineMeta<SplineOrder>& spline_meta,
                      double gyro_weight, double acce_weight,
                      double bias_weight) {
    using Functor = GyroAcceBiasFactor;
    return (new ceres::DynamicAutoDiffCostFunction<Functor>(
        new Functor(time_ns, imu_data, imu_bias, spline_meta, gyro_weight,
                    acce_weight, bias_weight)));
  }

 private:
  int64_t time_ns_;
  IMUData imu_data_;
  IMUBias imu_bias_;
  SplineMeta<SplineOrder> spline_meta_;
  double gyro_weight_;
  double acce_weight_;
  double bias_weight_;
  double inv_dt_;
};

class IMUPoseFactor {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  IMUPoseFactor(int64_t time_ns, const PoseData& pose_data,
                const SplineMeta<SplineOrder>& spline_meta, double pos_weight,
                double rot_weight)
      : time_ns_(time_ns),
        pose_data_(pose_data),
        spline_meta_(spline_meta),
        pos_weight_(pos_weight),
        rot_weight_(rot_weight) {
    inv_dt_ = 1e9 * 1.0 / spline_meta_.segments.begin()->dt_ns;
  }

  template <class T>
  bool operator()(T const* const* sKnots, T* sResiduals) const {
    size_t Knot_offset = 2 * spline_meta_.NumParameters();
    T time_offset_in_ns = sKnots[Knot_offset][0];
    T t_corrected = T(time_ns_) + time_offset_in_ns;

    size_t R_offset;  // should be zero if not estimate time offset
    size_t P_offset;
    T u;
    spline_meta_.ComputeSplineIndex(t_corrected, R_offset, u);
    P_offset = R_offset + spline_meta_.NumParameters();

    Sophus::SO3<T> R_IkToG;
    CeresSplineHelperJet<T, SplineOrder>::template evaluate_lie<Sophus::SO3>(
        sKnots + R_offset, u, inv_dt_, &R_IkToG);

    Eigen::Matrix<T, 3, 1> p_IkinG;
    CeresSplineHelperJet<T, SplineOrder>::template evaluate<3, 0>(
        sKnots + P_offset, u, inv_dt_, &p_IkinG);

    Eigen::Map<Eigen::Matrix<T, 6, 1>> residuals(sResiduals);
    residuals.template block<3, 1>(0, 0) =
        T(rot_weight_) * (R_IkToG * pose_data_.orientation.inverse()).log();
    residuals.template block<3, 1>(3, 0) =
        T(pos_weight_) * (p_IkinG - pose_data_.position);
    return true;
  }
  static auto* Create(int64_t time_ns, const PoseData& pose_data,
                      const SplineMeta<SplineOrder>& spline_meta,
                      double pos_weight, double rot_weight) {
    using Functor = IMUPoseFactor;
    return (new ceres::DynamicAutoDiffCostFunction<Functor>(
        new Functor(time_ns, pose_data, spline_meta, pos_weight, rot_weight)));
  }

 private:
  int64_t time_ns_;
  PoseData pose_data_;
  SplineMeta<SplineOrder> spline_meta_;
  double pos_weight_;
  double rot_weight_;
  double inv_dt_;
};

class IMUGlobalVelocityFactor {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  IMUGlobalVelocityFactor(const int64_t time_ns,
                          const Eigen::Vector3d& velocity,
                          const SplineMeta<SplineOrder>& spline_meta,
                          double vel_weight)
      : time_ns_(time_ns),
        velocity_(velocity),
        spline_meta_(spline_meta),
        vel_weight_(vel_weight) {
    inv_dt_ = 1e9 * 1.0 / spline_meta_.segments.begin()->dt_ns;
  }

  template <class T>
  bool operator()(T const* const* sKnots, T* sResiduals) const {
    using Vec3T = Eigen::Matrix<T, 3, 1>;

    size_t Knot_offset = spline_meta_.NumParameters();
    T time_offset_in_ns = sKnots[Knot_offset][0];
    T t_corrected = T(time_ns_) + time_offset_in_ns;

    size_t P_offset;
    T u;
    spline_meta_.ComputeSplineIndex(t_corrected, P_offset, u);

    Vec3T v_IkinG;
    CeresSplineHelperJet<T, SplineOrder>::template evaluate<3, 1>(
        sKnots + P_offset, u, inv_dt_, &v_IkinG);

    Eigen::Map<Vec3T> residuals(sResiduals);

    residuals = T(vel_weight_) * (v_IkinG - velocity_.template cast<T>());
    return true;
  }

  static auto* Create(const int64_t time_ns, const Eigen::Vector3d& velocity,
                      const SplineMeta<SplineOrder>& spline_meta,
                      double vel_weight) {
    using Functor = IMUGlobalVelocityFactor;
    return (new ceres::DynamicAutoDiffCostFunction<Functor>(
        new Functor(time_ns, velocity, spline_meta, vel_weight)));
  }

 private:
  int64_t time_ns_;
  Eigen::Vector3d velocity_;
  SplineMeta<SplineOrder> spline_meta_;
  double vel_weight_;
  double inv_dt_;
};


class IMULocalVelocityFactor {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  IMULocalVelocityFactor(const int64_t time_ns,
                         const Eigen::Vector3d& local_velocity,
                         const SplineMeta<SplineOrder>& spline_meta,
                         double weight)
      : time_ns_(time_ns),
        local_velocity_(local_velocity),
        spline_meta_(spline_meta),
        weight_(weight) {
    inv_dt_ = 1e9 * 1.0 / spline_meta_.segments.begin()->dt_ns;
  }

  template <class T>
  bool operator()(T const* const* sKnots, T* sResiduals) const {
    using Vec3T = Eigen::Matrix<T, 3, 1>;

    size_t Knot_offset = 2 * spline_meta_.NumParameters();
    T time_offset_in_ns = sKnots[Knot_offset][0];
    T t_corrected = T(time_ns_) + time_offset_in_ns;

    size_t R_offset;  // should be zero if not estimate time offset
    size_t P_offset;
    T u;
    spline_meta_.ComputeSplineIndex(t_corrected, R_offset, u);
    P_offset = R_offset + spline_meta_.NumParameters();

    Sophus::SO3<T> R_IkToG;
    CeresSplineHelperJet<T, SplineOrder>::template evaluate_lie<Sophus::SO3>(
        sKnots + R_offset, u, inv_dt_, &R_IkToG);

    Vec3T v_IkinG;
    CeresSplineHelperJet<T, SplineOrder>::template evaluate<3, 1>(
        sKnots + P_offset, u, inv_dt_, &v_IkinG);

    Eigen::Map<Vec3T> residuals(sResiduals);
    residuals = R_IkToG * local_velocity_.template cast<T>() - v_IkinG;
    residuals = (T(weight_) * residuals).eval();
    return true;
  }

  static auto* Create(int64_t time_ns, const Eigen::Vector3d& local_velocity,
                      const SplineMeta<SplineOrder>& spline_meta,
                      double weight) {
    using Functor = IMULocalVelocityFactor;
    return (new ceres::DynamicAutoDiffCostFunction<Functor>(
        new Functor(time_ns, local_velocity, spline_meta, weight)));
  }

 private:
  int64_t time_ns_;
  Eigen::Vector3d local_velocity_;
  SplineMeta<SplineOrder> spline_meta_;
  double inv_dt_;
  double weight_;
};

class VelocityConstraintFactor {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  VelocityConstraintFactor(const int64_t time_ns,
                           const Eigen::Matrix<double, 6, 1>& local_gyro_vel,
                           const SplineMeta<SplineOrder>& spline_meta,
                           double gyro_weight, double velocity_weight)
      : time_ns_(time_ns),
        local_gyro_vel_(local_gyro_vel),
        spline_meta_(spline_meta),
        gyro_weight_(gyro_weight),
        velocity_weight_(velocity_weight) {
    inv_dt_ = 1e9 * 1.0 / spline_meta_.segments.begin()->dt_ns;
  }

  template <class T>
  bool operator()(T const* const* sKnots, T* sResiduals) const {
    using Vec3T = Eigen::Matrix<T, 3, 1>;
    using Vec6T = Eigen::Matrix<T, 6, 1>;
    using SO3T = Sophus::SO3<T>;
    using Tangent = typename Sophus::SO3<T>::Tangent;

    size_t Knot_offset = 2 * spline_meta_.NumParameters();
    T time_offset_in_ns = sKnots[Knot_offset][0];
    T t_corrected = T(time_ns_) + time_offset_in_ns;

    size_t R_offset;  // should be zero if not estimate time offset
    size_t P_offset;
    T u;
    spline_meta_.ComputeSplineIndex(t_corrected, R_offset, u);
    P_offset = R_offset + spline_meta_.NumParameters();

    SO3T R_w_i;
    Tangent rot_vel;
    CeresSplineHelperJet<T, SplineOrder>::template evaluate_lie<Sophus::SO3>(
        sKnots + R_offset, u, inv_dt_, &R_w_i, &rot_vel);

    Vec3T vel_w;
    CeresSplineHelperJet<T, SplineOrder>::template evaluate<3, 1>(
        sKnots + P_offset, u, inv_dt_, &vel_w);

    Vec6T local_gyro_vel_T = local_gyro_vel_.template cast<T>();

    Vec3T gyro_residuals = rot_vel - local_gyro_vel_T.head(3);
    Vec3T vel_residuals = R_w_i.inverse() * vel_w - local_gyro_vel_T.tail(3);

    Eigen::Map<Vec6T> residuals(sResiduals);
    residuals.template block<3, 1>(0, 0) = T(gyro_weight_) * gyro_residuals;
    residuals.template block<3, 1>(3, 0) = T(velocity_weight_) * vel_residuals;

    return true;
  }

  static auto* Create(int64_t time_ns,
                      const Eigen::Matrix<double, 6, 1>& local_gyro_vel,
                      const SplineMeta<SplineOrder>& spline_meta,
                      double gyro_weight, double velocity_weight) {
    using Functor = VelocityConstraintFactor;
    return (new ceres::DynamicAutoDiffCostFunction<Functor>(new Functor(
        time_ns, local_gyro_vel, spline_meta, gyro_weight, velocity_weight)));
  }

 private:
  int64_t time_ns_;
  Eigen::Matrix<double, 6, 1> local_gyro_vel_;
  SplineMeta<SplineOrder> spline_meta_;
  double gyro_weight_;
  double velocity_weight_;
  double inv_dt_;
};

class LiDARPoseFactor {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  LiDARPoseFactor(const int64_t time_ns, const PoseData& pose_data,
                  const SplineMeta<SplineOrder>& spline_meta, double pos_weight,
                  double rot_weight)
      : time_ns_(time_ns),
        pose_data_(pose_data),
        spline_meta_(spline_meta),
        pos_weight_(pos_weight),
        rot_weight_(rot_weight) {
    inv_dt_ = 1e9 * 1.0 / spline_meta_.segments.begin()->dt_ns;
  }

  template <class T>
  bool operator()(T const* const* sKnots, T* sResiduals) const {
    using Vec3T = Eigen::Matrix<T, 3, 1>;
    using Vec6T = Eigen::Matrix<T, 6, 1>;
    using SO3T = Sophus::SO3<T>;

    size_t R_offset;  // should be zero if not estimate time offset
    size_t P_offset;
    double u;
    spline_meta_.ComputeSplineIndex(time_ns_, R_offset, u);
    P_offset = R_offset + spline_meta_.NumParameters();

    SO3T R_IkToG;
    CeresSplineHelper<SplineOrder>::template evaluate_lie<T, Sophus::SO3>(
        sKnots + R_offset, u, inv_dt_, &R_IkToG);

    Vec3T p_IkinG;
    CeresSplineHelper<SplineOrder>::template evaluate<T, 3, 0>(
        sKnots + P_offset, u, inv_dt_, &p_IkinG);

    int Kont_offset = 2 * spline_meta_.NumParameters();
    Eigen::Map<SO3T const> const R_LtoI(sKnots[Kont_offset]);
    Eigen::Map<Vec3T const> const p_LinI(sKnots[Kont_offset + 1]);

    SO3T R_LkToG = R_IkToG * R_LtoI;
    Vec3T p_LkinG = R_IkToG * p_LinI + p_IkinG;

    Eigen::Map<Vec6T> residuals(sResiduals);
    residuals.template block<3, 1>(0, 0) =
        T(rot_weight_) * (R_LkToG * pose_data_.orientation.inverse()).log();

    residuals.template block<3, 1>(3, 0) =
        T(pos_weight_) * (p_LkinG - pose_data_.position);

    return true;
  }

  static auto* Create(int64_t time_ns, const PoseData& pose_data,
                      const SplineMeta<SplineOrder>& spline_meta,
                      double pos_weight, double rot_weight) {
    using Functor = LiDARPoseFactor;
    return (new ceres::DynamicAutoDiffCostFunction<Functor>(
        new Functor(time_ns, pose_data, spline_meta, pos_weight, rot_weight)));
  }

 private:
  int64_t time_ns_;
  PoseData pose_data_;
  SplineMeta<SplineOrder> spline_meta_;
  double pos_weight_;
  double rot_weight_;
  double inv_dt_;
};

class PreIntegrationFactor {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  PreIntegrationFactor(IntegrationBase* pre_integration, int64_t ti_ns,
                       int64_t tj_ns, SplineMeta<SplineOrder> spline_meta)
      : pre_integration_(pre_integration),
        t_i_ns_(ti_ns),
        t_j_ns_(tj_ns),
        spline_meta_(spline_meta) {
    inv_dt_ = 1e9 * 1.0 / spline_meta_.segments.begin()->dt_ns;
  }

  template <class T>
  bool operator()(T const* const* sKnots, T* sResiduals) const {
    using Vec3T = Eigen::Matrix<T, 3, 1>;
    using Vec15T = Eigen::Matrix<T, 15, 1>;
    using Mat15T = Eigen::Matrix<T, 15, 15>;
    using SO3T = Sophus::SO3<T>;

    //旋转控制点+平移控制点
    size_t R_offset_i, R_offset_j;
    size_t P_offset_i, P_offset_j;
    double u_i, u_j;
    spline_meta_.ComputeSplineIndex(t_i_ns_, R_offset_i, u_i);
    spline_meta_.ComputeSplineIndex(t_j_ns_, R_offset_j, u_j);
    P_offset_i = R_offset_i + spline_meta_.NumParameters();
    P_offset_j = R_offset_j + spline_meta_.NumParameters();

    SO3T Rwbi, Rwbj;
    Vec3T Pbi, Pbj;
    Vec3T Vbi, Vbj;
    CeresSplineHelper<SplineOrder>::template evaluate_lie<T, Sophus::SO3>(
        sKnots + R_offset_i, u_i, inv_dt_, &Rwbi);
    CeresSplineHelper<SplineOrder>::template evaluate_lie<T, Sophus::SO3>(
        sKnots + R_offset_j, u_j, inv_dt_, &Rwbj);
    CeresSplineHelper<SplineOrder>::template evaluate<T, 3, 0>(
        sKnots + P_offset_i, u_i, inv_dt_, &Pbi);
    CeresSplineHelper<SplineOrder>::template evaluate<T, 3, 0>(
        sKnots + P_offset_j, u_j, inv_dt_, &Pbj);
    CeresSplineHelper<SplineOrder>::template evaluate<T, 3, 1>(
        sKnots + P_offset_i, u_i, inv_dt_, &Vbi);
    CeresSplineHelper<SplineOrder>::template evaluate<T, 3, 1>(
        sKnots + P_offset_j, u_j, inv_dt_, &Vbj);

    size_t Knot_offset = 2 * spline_meta_.NumParameters();
    Eigen::Map<Vec3T const> gyro_bias_i_(sKnots[Knot_offset]);
    Eigen::Map<Vec3T const> gyro_bias_j_(sKnots[Knot_offset + 1]);
    Eigen::Map<Vec3T const> accel_bias_i_(sKnots[Knot_offset + 2]);
    Eigen::Map<Vec3T const> accel_bias_j_(sKnots[Knot_offset + 3]);

    Vec3T Bgi = gyro_bias_i_;
    Vec3T Bgj = gyro_bias_j_;
    Vec3T Bai = accel_bias_i_;
    Vec3T Baj = accel_bias_j_;

    Eigen::Map<Vec15T> residual(sResiduals);
    residual = pre_integration_->evaluate333(Pbi, Rwbi, Vbi, Bai, Bgi, Pbj,
                                             Rwbj, Vbj, Baj, Bgj);
    Mat15T cov_temp = pre_integration_->covariance.template cast<T>();
    Mat15T sqrt_info =
        Eigen::LLT<Mat15T>(cov_temp.inverse()).matrixL().transpose();
    residual = sqrt_info * residual;
    return true;
  }

  static auto* Create(IntegrationBase* pre_integration, int64_t ti_ns,
                      int64_t tj_ns, SplineMeta<SplineOrder> spline_meta) {
    using Functor = PreIntegrationFactor;
    return (new ceres::DynamicAutoDiffCostFunction<Functor>(
        new Functor(pre_integration, ti_ns, tj_ns, spline_meta)));
  }

 private:
  IntegrationBase* pre_integration_;
  int64_t t_i_ns_;
  int64_t t_j_ns_;
  SplineMeta<SplineOrder> spline_meta_;
  double inv_dt_;
};

class RelativeOrientationFactor {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  RelativeOrientationFactor(const SO3d S_BtoA, int64_t ta_ns, int64_t tb_ns,
                            SplineMeta<SplineOrder> spline_meta)
      : S_BtoA_(S_BtoA),
        ta_ns_(ta_ns),
        tb_ns_(tb_ns),
        spline_meta_(spline_meta) {
    inv_dt_ = 1e9 * 1.0 / spline_meta_.segments.begin()->dt_ns;
  }

  template <class T>
  bool operator()(T const* const* sKnots, T* sResiduals) const {
    using Vec3T = Eigen::Matrix<T, 3, 1>;
    using SO3T = Sophus::SO3<T>;

    size_t R_offset[2];
    double u[2];
    spline_meta_.ComputeSplineIndex(ta_ns_, R_offset[0], u[0]);
    spline_meta_.ComputeSplineIndex(tb_ns_, R_offset[1], u[1]);

    SO3T Ra, Rb;
    CeresSplineHelper<SplineOrder>::template evaluate_lie<T, Sophus::SO3>(
        sKnots + R_offset[0], u[0], inv_dt_, &Ra);
    CeresSplineHelper<SplineOrder>::template evaluate_lie<T, Sophus::SO3>(
        sKnots + R_offset[1], u[1], inv_dt_, &Rb);

    ///计算残差
    Eigen::Map<Vec3T> residual(sResiduals);
    residual.template block<3, 1>(0, 0) = (Rb.inverse() * Ra * S_BtoA_).log();

    residual.applyOnTheLeft(sqrt_info.template cast<T>());

    return true;
  }

  static auto* Create(const SO3d S_BtoA, int64_t ta_ns, int64_t tb_ns,
                      SplineMeta<SplineOrder> spline_meta) {
    using Functor = RelativeOrientationFactor;
    return (new ceres::DynamicAutoDiffCostFunction<Functor>(
        new Functor(S_BtoA, ta_ns, tb_ns, spline_meta)));
  }

  static inline Eigen::Matrix3d sqrt_info;

 private:
  SO3d S_BtoA_;
  int64_t ta_ns_, tb_ns_;
  SplineMeta<SplineOrder> spline_meta_;
  double inv_dt_;
};

}  // namespace auto_diff
}  // namespace clic