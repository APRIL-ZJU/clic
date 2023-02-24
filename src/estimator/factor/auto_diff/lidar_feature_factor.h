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
#include <lidar_odometry/lidar_feature.h>
#include <spline/ceres_spline_helper_jet.h>
#include <spline/spline_segment.h>
#include <utils/parameter_struct.h>

#include <sophus_lib/so3.hpp>

namespace clic {
namespace auto_diff {
class PointFeatureFactor {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  PointFeatureFactor(int64_t t_point_ns, int64_t t_map_ns,
                     const PointCorrespondence& pc,
                     const SplineMeta<SplineOrder>& spline_meta, double weight)
      : t_point_ns_(t_point_ns),
        t_map_ns_(t_map_ns),
        measurement_(pc),
        spline_meta_(spline_meta),
        weight_(weight) {
    inv_dt_ = 1e9 * 1.0 / spline_meta_.segments.begin()->dt_ns;
  }

  template <class T>
  bool operator()(T const* const* sKnots, T* sResiduals) const {
    using SO3T = Sophus::SO3<T>;
    using Vec3T = Eigen::Matrix<T, 3, 1>;

    size_t Knot_offset = 2 * spline_meta_.NumParameters();
    Eigen::Map<SO3T const> const R_LtoI(sKnots[Knot_offset]);
    Eigen::Map<Vec3T const> const p_LinI(sKnots[Knot_offset + 1]);
    T t_offset = sKnots[Knot_offset + 2][0];

    T t[2];
    t[0] = T(t_map_ns_) + t_offset;
    t[1] = T(t_point_ns_) + t_offset;

    T u[2];
    size_t R_offset[2];
    size_t P_offset[2];
    spline_meta_.ComputeSplineIndex(t[0], R_offset[0], u[0]);
    P_offset[0] = R_offset[0] + spline_meta_.NumParameters();

    spline_meta_.ComputeSplineIndex(t[1], R_offset[1], u[1]);
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

    SO3T R_IkToI0 = R_IkToG[0].inverse() * R_IkToG[1];
    Vec3T p_IkinI0 = R_IkToG[0].inverse() * (p_IkinG[1] - p_IkinG[0]);

    // {M} frame is coinside with {L0) frame.
    Vec3T p_Lk = measurement_.point.template cast<T>();
    Vec3T p_Ik = R_LtoI * p_Lk + p_LinI;
    Vec3T p_I0 = R_IkToI0 * p_Ik + p_IkinI0;
    Vec3T p_L0 = R_LtoI.inverse() * (p_I0 - p_LinI);

    T dist;
    if (measurement_.geo_type == GeometryType::Plane) {
      Vec3T norm = (measurement_.geo_plane.template cast<T>()).head(3);
      dist = p_L0.dot(norm) + T(measurement_.geo_plane[3]);
    } else {
      // omit item 1 =: 1.0 / measurement_.geo_normal.norm()
      dist = (p_L0 - measurement_.geo_point.template cast<T>())
                 .cross(measurement_.geo_normal.template cast<T>())
                 .norm();
    }

    Eigen::Map<Eigen::Matrix<T, 1, 1>> residuals(sResiduals);
    residuals.template block<1, 1>(0, 0) = Eigen::Matrix<T, 1, 1>(dist);

    residuals = T(weight_) * residuals;

    return true;
  }

  static auto* Create(int64_t t_point_ns, int64_t t_map_ns,
                      const PointCorrespondence& pc,
                      const SplineMeta<SplineOrder>& spline_meta,
                      double weight) {
    using Functor = PointFeatureFactor;
    return (new ceres::DynamicAutoDiffCostFunction<Functor>(
        new Functor(t_point_ns, t_map_ns, pc, spline_meta, weight)));
  }

 private:
  int64_t t_point_ns_;
  int64_t t_map_ns_;
  PointCorrespondence measurement_;
  SplineMeta<SplineOrder> spline_meta_;
  double weight_;
  double inv_dt_;
};

class LoamFeatureOptMapPoseFactor {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  LoamFeatureOptMapPoseFactor(int64_t t_point_ns, const PointCorrespondence& pc,
                              const SplineMeta<SplineOrder>& spline_meta,
                              double weight)
      : t_point_ns_(t_point_ns),
        measurement_(pc),
        spline_meta_(spline_meta),
        weight_(weight) {
    assert(init_flag && "LoamFeatureOptMapPoseFactor [auto] not init param");
    inv_dt_ = 1e9 * 1.0 / spline_meta_.segments.begin()->dt_ns;
  }

  template <class T>
  bool operator()(T const* const* sKnots, T* sResiduals) const {
    using SO3T = Sophus::SO3<T>;
    using Vec3T = Eigen::Matrix<T, 3, 1>;

    size_t Knot_offset = 2 * spline_meta_.NumParameters();

    T u;
    size_t R_offset;
    size_t P_offset;
    spline_meta_.ComputeSplineIndex(T(t_point_ns_), R_offset, u);
    P_offset = R_offset + spline_meta_.NumParameters();

    SO3T S_IkToG;
    CeresSplineHelperJet<T, SplineOrder>::template evaluate_lie<Sophus::SO3>(
        sKnots + R_offset, u, inv_dt_, &S_IkToG);

    Vec3T p_IkinG;
    CeresSplineHelperJet<T, SplineOrder>::template evaluate<3, 0>(
        sKnots + P_offset, u, inv_dt_, &p_IkinG);

    Eigen::Map<SO3T const> S_ImtoG(sKnots[Knot_offset]);
    Eigen::Map<Vec3T const> p_IminG(sKnots[Knot_offset + 1]);

    SO3T S_IkToI0 = S_ImtoG.inverse() * S_IkToG;
    Vec3T p_IkinI0 = S_ImtoG.inverse() * (p_IkinG - p_IminG);

    SO3T S_LtoI_T = S_LtoI.template cast<T>();
    Vec3T p_LinI_T = p_LinI.template cast<T>();

    // {M} frame is coinside with {L0) frame.
    Vec3T p_Lk = measurement_.point.template cast<T>();
    Vec3T p_Ik = S_LtoI_T * p_Lk + p_LinI_T;
    Vec3T p_I0 = S_IkToI0 * p_Ik + p_IkinI0;
    Vec3T p_L0 = S_LtoI_T.inverse() * (p_I0 - p_LinI_T);

    T dist;
    if (measurement_.geo_type == GeometryType::Plane) {
      Vec3T norm = (measurement_.geo_plane.template cast<T>()).head(3);
      dist = p_L0.dot(norm) + T(measurement_.geo_plane[3]);
    } else {
      // omit item 1 =: 1.0 / measurement_.geo_normal.norm()
      dist = (p_L0 - measurement_.geo_point.template cast<T>())
                 .cross(measurement_.geo_normal.template cast<T>())
                 .norm();
    }

    Eigen::Map<Eigen::Matrix<T, 1, 1>> residuals(sResiduals);
    residuals.template block<1, 1>(0, 0) = Eigen::Matrix<T, 1, 1>(dist);

    residuals = T(weight_) * residuals;

    return true;
  }

  static auto* Create(int64_t t_point_ns, const PointCorrespondence& pc,
                      const SplineMeta<SplineOrder>& spline_meta,
                      double weight) {
    using Functor = LoamFeatureOptMapPoseFactor;
    return (new ceres::DynamicAutoDiffCostFunction<Functor>(
        new Functor(t_point_ns, pc, spline_meta, weight)));
  }

  static void SetParam(Sophus::SO3<double> _S_LtoI, Eigen::Vector3d _p_LinI) {
    init_flag = true;
    S_LtoI = _S_LtoI;
    p_LinI = _p_LinI;
  }

 private:
  static inline bool init_flag = false;
  static inline Sophus::SO3<double> S_LtoI;
  static inline Eigen::Vector3d p_LinI;

  int64_t t_point_ns_;
  PointCorrespondence measurement_;
  SplineMeta<SplineOrder> spline_meta_;
  double weight_;
  double inv_dt_;
};

}  // namespace auto_diff
}  // namespace clic
