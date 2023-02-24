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
#include <spline/ceres_spline_helper_jet.h>
#include <spline/spline_segment.h>
#include <utils/parameter_struct.h>

#include <sophus_lib/so3.hpp>

namespace clic {
namespace auto_diff {

class ImageFeatureFactor {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ImageFeatureFactor(const int64_t t_i_ns, const Eigen::Vector3d& p_i,
                     const int64_t t_j_ns, const Eigen::Vector3d& p_j,
                     const SplineMeta<SplineOrder>& spline_meta)
      : t_i_ns_(t_i_ns),
        p_i_(p_i),
        t_j_ns_(t_j_ns),
        p_j_(p_j),
        spline_meta_(spline_meta) {
    assert(init_flag && "ImageFeatureFactor not init param");
    inv_dt_ = 1e9 * 1.0 / spline_meta_.segments.begin()->dt_ns;
  }

  template <class T>
  bool operator()(T const* const* sKnots, T* sResiduals) const {
    using SO3T = Sophus::SO3<T>;
    using Vec3T = Eigen::Matrix<T, 3, 1>;

    size_t Knot_offset = 2 * spline_meta_.NumParameters();
    T d_inv = sKnots[Knot_offset][0];
    T time_offset_in_ns = sKnots[Knot_offset + 1][0];
    T ti_corrected = T(t_i_ns_) + time_offset_in_ns;
    T tj_corrected = T(t_j_ns_) + time_offset_in_ns;

    T u[2];
    size_t R_offset[2];
    size_t P_offset[2];
    spline_meta_.ComputeSplineIndex(T(ti_corrected), R_offset[0], u[0]);
    P_offset[0] = R_offset[0] + spline_meta_.NumParameters();

    spline_meta_.ComputeSplineIndex(T(tj_corrected), R_offset[1], u[1]);
    P_offset[1] = R_offset[1] + spline_meta_.NumParameters();

    SO3T R_IkToG[2];
    CeresSplineHelperJet<T, SplineOrder>::template evaluate_lie<Sophus::SO3>(
        sKnots + R_offset[0], u[0], inv_dt_, &R_IkToG[0]);
    CeresSplineHelperJet<T, SplineOrder>::template evaluate_lie<Sophus::SO3>(
        sKnots + R_offset[1], u[1], inv_dt_, &R_IkToG[1]);

    Vec3T p_IkinG[2];
    CeresSplineHelperJet<T, SplineOrder>::template evaluate<3, 0>(
        sKnots + P_offset[0], u[0], inv_dt_, &p_IkinG[0]);
    CeresSplineHelperJet<T, SplineOrder>::template evaluate<3, 0>(
        sKnots + P_offset[1], u[1], inv_dt_, &p_IkinG[1]);

    SO3T R_Ii_To_Ij = R_IkToG[1].inverse() * R_IkToG[0];
    Vec3T p_Ii_in_Ij = R_IkToG[1].inverse() * (p_IkinG[0] - p_IkinG[1]);

    Vec3T p_CinI = this->p_CinI.template cast<T>();
    SO3T S_CtoI = this->S_CtoI.template cast<T>();

    SO3T R_Ci_To_Cj = S_CtoI.inverse() * R_Ii_To_Ij * S_CtoI;
    Vec3T p_Ci_in_Cj =
        S_CtoI.inverse() * (R_Ii_To_Ij * p_CinI + p_Ii_in_Ij - p_CinI);

    Vec3T x_ci = p_i_.template cast<T>() / d_inv;
    Vec3T x_cj = R_Ci_To_Cj * x_ci + p_Ci_in_Cj;

    T depth_j = x_cj.z();
    Vec3T error = x_cj / depth_j - p_j_.template cast<T>();

    Eigen::Map<Eigen::Matrix<T, 2, 1>> residual(sResiduals);
    residual = error.template block<2, 1>(0, 0);
    residual.applyOnTheLeft(sqrt_info.template cast<T>());

    return true;
  }

  static auto* Create(const int64_t t_i_ns, const Eigen::Vector3d& p_i,
                      const int64_t t_j_ns, const Eigen::Vector3d& p_j,
                      const SplineMeta<SplineOrder>& spline_meta) {
    using Functor = ImageFeatureFactor;
    return (new ceres::DynamicAutoDiffCostFunction<Functor>(
        new Functor(t_i_ns, p_i, t_j_ns, p_j, spline_meta)));
  }

  static void SetParam(Sophus::SO3<double> _S_CtoI, Eigen::Vector3d _p_CinI) {
    init_flag = true;
    S_CtoI = _S_CtoI;
    p_CinI = _p_CinI;
  }

  // double focal_length = 450.;
  static inline Eigen::Matrix2d sqrt_info =
      450. / 1.5 * Eigen::Matrix2d::Identity();

 private:
  static inline bool init_flag = false;
  static inline Sophus::SO3<double> S_CtoI;
  static inline Eigen::Vector3d p_CinI;

  int64_t t_i_ns_;
  Eigen::Vector3d p_i_;
  int64_t t_j_ns_;
  Eigen::Vector3d p_j_;

  SplineMeta<SplineOrder> spline_meta_;
  double inv_dt_;
};

class Image3D2DFactor {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Image3D2DFactor(const int64_t t_j_ns, const Eigen::Vector3d& p_j,
                  const SplineMeta<SplineOrder>& spline_meta)
      : t_j_ns_(t_j_ns), p_j_(p_j), spline_meta_(spline_meta) {
    assert(init_flag && "Image3D2DFactor not init param");
    inv_dt_ = 1e9 * 1.0 / spline_meta_.segments.begin()->dt_ns;
  }

  template <class T>
  bool operator()(T const* const* sKnots, T* sResiduals) const {
    using SO3T = Sophus::SO3<T>;
    using Vec3T = Eigen::Matrix<T, 3, 1>;

    size_t Knot_offset = 2 * spline_meta_.NumParameters();
    Eigen::Map<const Vec3T> p_G(sKnots[Knot_offset]);
    T time_offset_in_ns = sKnots[Knot_offset + 1][0];
    T tj_corrected = T(t_j_ns_) + time_offset_in_ns;

    T u;
    size_t R_offset, P_offset;
    spline_meta_.ComputeSplineIndex(T(tj_corrected), R_offset, u);
    P_offset = R_offset + spline_meta_.NumParameters();

    SO3T S_IjToG;
    CeresSplineHelperJet<T, SplineOrder>::template evaluate_lie<Sophus::SO3>(
        sKnots + R_offset, u, inv_dt_, &S_IjToG);
    Vec3T p_IjinG;
    CeresSplineHelperJet<T, SplineOrder>::template evaluate<3, 0>(
        sKnots + P_offset, u, inv_dt_, &p_IjinG);

    Vec3T p_CinI = this->p_CinI.template cast<T>();
    SO3T S_CtoI = this->S_CtoI.template cast<T>();

    SO3T S_GtoCj = (S_IjToG * S_CtoI).inverse();
    Vec3T x_cj = S_GtoCj * (p_G - p_IjinG) - S_CtoI.inverse() * p_CinI;

    T depth_j = x_cj.z();
    Vec3T error = x_cj / depth_j - p_j_.template cast<T>();

    Eigen::Map<Eigen::Matrix<T, 2, 1>> residual(sResiduals);
    residual = error.template block<2, 1>(0, 0);
    residual.applyOnTheLeft(sqrt_info.template cast<T>());

    return true;
  }

  static auto* Create(const int64_t t_j_ns, const Eigen::Vector3d& p_j,
                      const SplineMeta<SplineOrder>& spline_meta) {
    using Functor = Image3D2DFactor;
    return (new ceres::DynamicAutoDiffCostFunction<Functor>(
        new Functor(t_j_ns, p_j, spline_meta)));
  }

  static void SetParam(Sophus::SO3<double> _S_CtoI, Eigen::Vector3d _p_CinI) {
    init_flag = true;
    S_CtoI = _S_CtoI;
    p_CinI = _p_CinI;
  }

  // double focal_length = 450.;
  static inline Eigen::Matrix2d sqrt_info =
      450. / 1.5 * Eigen::Matrix2d::Identity();

 private:
  static inline bool init_flag = false;
  static inline Sophus::SO3<double> S_CtoI;
  static inline Eigen::Vector3d p_CinI;

  int64_t t_j_ns_;
  Eigen::Vector3d p_j_;

  SplineMeta<SplineOrder> spline_meta_;
  double inv_dt_;
};

class ImageFeatureOnePoseFactor {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

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
    assert(init_flag && "ImageFeatureFactor not init param");
    inv_dt_ = 1e9 * 1.0 / spline_meta_.segments.begin()->dt_ns;
  }

  template <class T>
  bool operator()(T const* const* sKnots, T* sResiduals) const {
    using SO3T = Sophus::SO3<T>;
    using Vec3T = Eigen::Matrix<T, 3, 1>;

    size_t Knot_offset = 2 * spline_meta_.NumParameters();
    T d_inv = sKnots[Knot_offset][0];

    T u;
    size_t R_offset, P_offset;
    spline_meta_.ComputeSplineIndex(T(t_j_ns_), R_offset, u);
    P_offset = R_offset + spline_meta_.NumParameters();

    SO3T S_IkToG;
    CeresSplineHelperJet<T, SplineOrder>::template evaluate_lie<Sophus::SO3>(
        sKnots + R_offset, u, inv_dt_, &S_IkToG);
    Vec3T p_IkinG;
    CeresSplineHelperJet<T, SplineOrder>::template evaluate<3, 0>(
        sKnots + P_offset, u, inv_dt_, &p_IkinG);

    SO3T S_Ii_To_Ij = S_IkToG.inverse() * S_IitoG_;
    Vec3T p_Ii_in_Ij = S_IkToG.inverse() * (p_IiinG_ - p_IkinG);

    Vec3T p_CinI = this->p_CinI.template cast<T>();
    SO3T S_CtoI = this->S_CtoI.template cast<T>();

    SO3T S_Ci_To_Cj = S_CtoI.inverse() * S_Ii_To_Ij * S_CtoI;
    Vec3T p_Ci_in_Cj =
        S_CtoI.inverse() * (S_Ii_To_Ij * p_CinI + p_Ii_in_Ij - p_CinI);

    Vec3T x_ci = p_i_.template cast<T>() / d_inv;
    Vec3T x_cj = S_Ci_To_Cj * x_ci + p_Ci_in_Cj;

    T depth_j = x_cj.z();
    Vec3T error = x_cj / depth_j - p_j_.template cast<T>();

    Eigen::Map<Eigen::Matrix<T, 2, 1>> residual(sResiduals);
    residual = error.template block<2, 1>(0, 0);
    residual.applyOnTheLeft(sqrt_info.template cast<T>());

    return true;
  }

  static auto* Create(const Eigen::Vector3d& p_i, const SO3d& S_IitoG,
                      const Eigen::Vector3d& p_IiinG, const int64_t t_j_ns,
                      const Eigen::Vector3d& p_j,
                      const SplineMeta<SplineOrder>& spline_meta) {
    using Functor = ImageFeatureOnePoseFactor;
    return (new ceres::DynamicAutoDiffCostFunction<Functor>(
        new Functor(p_i, S_IitoG, p_IiinG, t_j_ns, p_j, spline_meta)));
  }

  static void SetParam(Sophus::SO3<double> _S_CtoI, Eigen::Vector3d _p_CinI) {
    init_flag = true;
    S_CtoI = _S_CtoI;
    p_CinI = _p_CinI;
  }

  // double focal_length = 450.;
  static inline Eigen::Matrix2d sqrt_info =
      450. / 1.5 * Eigen::Matrix2d::Identity();

 private:
  static inline bool init_flag = false;
  static inline Sophus::SO3<double> S_CtoI;
  static inline Eigen::Vector3d p_CinI;

  Eigen::Vector3d p_i_;
  SO3d S_IitoG_;
  Eigen::Vector3d p_IiinG_;

  int64_t t_j_ns_;
  Eigen::Vector3d p_j_;

  SplineMeta<SplineOrder> spline_meta_;
  double inv_dt_;
};

// linear factor
class ImageDepthFactor {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ImageDepthFactor(const Eigen::Vector3d& p_i, const Eigen::Vector3d& p_j,
                   const SO3d& S_CitoCj, const Eigen::Vector3d& p_CiinCj)
      : p_i_(p_i), p_j_(p_j), S_CitoCj_(S_CitoCj), p_CiinCj_(p_CiinCj) {}

  template <class T>
  bool operator()(T const* const* sKnots, T* sResiduals) const {
    using Vec3T = Eigen::Matrix<T, 3, 1>;
    using SO3T = Sophus::SO3<T>;

    T d_inv = sKnots[0][0];

    Vec3T p_Ci_in_Cj = this->p_CiinCj_.template cast<T>();
    SO3T S_Ci_To_Cj = this->S_CitoCj_.template cast<T>();

    Vec3T x_ci = p_i_.template cast<T>() / d_inv;
    Vec3T x_cj = S_Ci_To_Cj * x_ci + p_Ci_in_Cj;

    T depth_j = x_cj.z();
    Vec3T error = x_cj / depth_j - p_j_.template cast<T>();

    Eigen::Map<Eigen::Matrix<T, 2, 1>> residual(sResiduals);
    residual = error.template block<2, 1>(0, 0);
    residual.applyOnTheLeft(sqrt_info.template cast<T>());

    return true;
  }

  static auto* Create(const Eigen::Vector3d& p_i, const Eigen::Vector3d& p_j,
                      const SO3d& S_CitoCj, const Eigen::Vector3d& p_CiinCj) {
    using Functor = ImageDepthFactor;
    return (new ceres::DynamicAutoDiffCostFunction<Functor>(
        new Functor(p_i, p_j, S_CitoCj, p_CiinCj)));
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

class EpipolarFactor {
 public:
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
    inv_dt_ = 1e9 * 1.0 / spline_meta_.segments.begin()->dt_ns;
  }

  template <class T>
  bool operator()(T const* const* sKnots, T* sResiduals) const {
    using SO3T = Sophus::SO3<T>;
    using Vec3T = Eigen::Matrix<T, 3, 1>;
    using Mat3T = Eigen::Matrix<T, 3, 3>;

    T u;
    size_t R_offset;
    size_t P_offset;
    spline_meta_.ComputeSplineIndex(T(t_i_ns_), R_offset, u);
    P_offset = R_offset + spline_meta_.NumParameters();

    SO3T S_IitoG;
    Vec3T p_IiinG = Vec3T::Zero();
    CeresSplineHelperJet<T, SplineOrder>::template evaluate_lie<Sophus::SO3>(
        sKnots + R_offset, u, inv_dt_, &S_IitoG);
    CeresSplineHelperJet<T, SplineOrder>::template evaluate<3, 0>(
        sKnots + P_offset, u, inv_dt_, &p_IiinG);

    Vec3T x_i_T = x_i_.template cast<T>();
    Vec3T x_k_T = x_k_.template cast<T>();
    Vec3T p_CkinG_T = p_CkinG_.template cast<T>();
    SO3T S_GtoCk_T = S_GtoCk_.template cast<T>();

    SO3T S_CtoI_T = S_CtoI.template cast<T>();
    Vec3T p_CinI_T = p_CinI.template cast<T>();

    // 对极约束
    Vec3T Rxi = S_GtoCk_T * S_IitoG * S_CtoI_T * x_i_T;
    Mat3T t_hat =
        SO3T::hat(S_GtoCk_T * (S_IitoG * p_CinI_T + p_IiinG - p_CkinG_T));

    Eigen::Map<Eigen::Matrix<T, 1, 1>> residuals(sResiduals);
    residuals = x_k_T.transpose() * t_hat * Rxi;
    residuals = T(weight_) * residuals;

    return true;
  }

  static auto* Create(const int64_t t_i_ns, const Eigen::Vector3d& x_i,
                      const Eigen::Vector3d& x_k, const SO3d& S_GtoCk,
                      const Eigen::Vector3d& p_CkinG,
                      const SplineMeta<SplineOrder>& spline_meta,
                      double weight) {
    using Functor = EpipolarFactor;
    return (new ceres::DynamicAutoDiffCostFunction<Functor>(
        new Functor(t_i_ns, x_i, x_k, S_GtoCk, p_CkinG, spline_meta, weight)));
  }

  static void SetParam(Sophus::SO3<double> _S_CtoI, Eigen::Vector3d _p_CinI) {
    init_flag = true;
    S_CtoI = _S_CtoI;
    p_CinI = _p_CinI;
  }

 private:
  static inline bool init_flag = false;
  static inline Sophus::SO3<double> S_CtoI;
  static inline Eigen::Vector3d p_CinI;

  int64_t t_i_ns_;
  Eigen::Vector3d x_i_;

  Eigen::Vector3d x_k_;
  SO3d S_GtoCk_;
  Eigen::Vector3d p_CkinG_;

  SplineMeta<SplineOrder> spline_meta_;
  double weight_;

  double inv_dt_;
};

}  // namespace auto_diff
}  // namespace clic
