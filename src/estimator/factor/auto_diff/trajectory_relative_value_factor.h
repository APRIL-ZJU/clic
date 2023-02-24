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

class IMUDeltaOrientationFactor {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  IMUDeltaOrientationFactor(int64_t t_ref_ns, int64_t t_cur_ns,
                            const IMUData& ref_pose_data,
                            const IMUData& pose_data,
                            const SplineMeta<SplineOrder>& spline_meta,
                            double rot_weight)
      : t_ref_ns_(t_ref_ns),
        t_cur_ns_(t_cur_ns),
        ref_pose_data_(ref_pose_data),
        pose_data_(pose_data),
        spline_meta_(spline_meta),
        rot_weight_(rot_weight) {
    inv_dt_ = 1e9 * 1.0 / spline_meta_.segments.begin()->dt_ns;
  }

  template <class T>
  bool operator()(T const* const* sKnots, T* sResiduals) const {
    using SO3T = Sophus::SO3<T>;
    using Tangent = typename Sophus::SO3<T>::Tangent;

    T t[2];
    t[0] = T(t_ref_ns_);
    t[1] = T(t_cur_ns_);

    T u[2];
    size_t R_offset[2];

    spline_meta_.ComputeSplineIndex(t[0], R_offset[0], u[0]);
    spline_meta_.ComputeSplineIndex(t[1], R_offset[1], u[1]);

    SO3T R_IkToG[2];
    CeresSplineHelperJet<T, SplineOrder>::template evaluate_lie<Sophus::SO3>(
        sKnots + R_offset[0], u[0], inv_dt_, &R_IkToG[0]);
    CeresSplineHelperJet<T, SplineOrder>::template evaluate_lie<Sophus::SO3>(
        sKnots + R_offset[1], u[1], inv_dt_, &R_IkToG[1]);

    SO3T delta_rot_est = R_IkToG[0].inverse() * R_IkToG[1];
    SO3T delta_rot_mea =
        ref_pose_data_.orientation.inverse() * pose_data_.orientation;

    Eigen::Map<Tangent> residuals(sResiduals);

    residuals =
        T(rot_weight_) * ((delta_rot_mea.inverse() * delta_rot_est).log());
    return true;
  }
  static auto* Create(int64_t t_ref_ns, int64_t t_cur_ns,
                      const IMUData& ref_pose_data, const IMUData& pose_data,
                      const SplineMeta<SplineOrder>& spline_meta,
                      double rot_weight) {
    using Functor = IMUDeltaOrientationFactor;
    return (new ceres::DynamicAutoDiffCostFunction<Functor>(
        new Functor(t_ref_ns, t_cur_ns, ref_pose_data, pose_data, spline_meta,
                    rot_weight)));
  }

 private:
  int64_t t_ref_ns_;
  int64_t t_cur_ns_;
  IMUData ref_pose_data_;
  IMUData pose_data_;
  SplineMeta<SplineOrder> spline_meta_;
  double rot_weight_;
  double inv_dt_;
};

}  // namespace clic