#include "feature_manager.h"
#include <glog/logging.h>

using namespace std;

namespace clic {

void FeatureManager::clearState() { feature.clear(); }

int FeatureManager::getFeatureCount() const {
  int cnt = 0;
  for (auto &it : feature) {
    if (isLandmarkCandidate(it)) cnt++;
  }
  return cnt;
}

/// frame_count: 滑窗 ID
/// image: [feature_id, camera_id, (x,y,z,u,v,vx,vy)]
bool FeatureManager::addFeatureCheckParallax(
    int frame_count, const map<int, vector<pair<int, Vector7d>>> &image,
    double td) {
  ROS_DEBUG("input feature: %d", (int)image.size());
  ROS_DEBUG("num of feature: %d", getFeatureCount());
  double parallax_sum = 0;  // parallaxs of the latest second and third frame
  int parallax_num = 0;     // feature number of the latest second and third frame
  last_track_num = 0;       // feature track num of latest frame

  for (auto &id_pts : image) {
    FeaturePerFrame f_per_fra(id_pts.second[0].second, td);

    int feature_id = id_pts.first;
    auto it = find_if(feature.begin(), feature.end(),
                      [feature_id](const FeaturePerId &it) {
                        return it.feature_id == feature_id;
                      });

    if (it == feature.end()) {
      // add new feature, start_frame is current frame (frame_count)
      feature.push_back(FeaturePerId(feature_id, frame_count));
      feature.back().feature_per_frame.push_back(f_per_fra);
    } else if (it->feature_id == feature_id) {
      // we track that feature again 
      it->feature_per_frame.push_back(f_per_fra);
      last_track_num++;
    }
  }

  // chack if the latest second frame is a keyframe
  if (frame_count < 2 || last_track_num < 20) return true;

  for (auto &it_per_id : feature) {
    if (it_per_id.start_frame <= frame_count - 2 &&
        it_per_id.start_frame + int(it_per_id.feature_per_frame.size()) - 1 >=
            frame_count - 1) {
      parallax_sum += compensatedParallax2(it_per_id, frame_count);
      parallax_num++;
    }
  }

  if (parallax_num == 0) {
    return true;
  } else {
    LOG(INFO) << "parallax_sum: " << parallax_sum
              << "; parallax_num: " << parallax_num;
    LOG(INFO) << "current parallax: " << parallax_sum / parallax_num
              << "; MIN_PARALLAX: " << MIN_PARALLAX;
    return parallax_sum / parallax_num >= MIN_PARALLAX;
  }
}

vector<pair<Vector3d, Vector3d>> FeatureManager::getCorresponding(
    int frame_count_l, int frame_count_r, std::vector<int> *fea_ids) const {
  vector<pair<Vector3d, Vector3d>> corres;
  for (auto &it : feature) {
    if (it.start_frame <= frame_count_l && it.endFrame() >= frame_count_r) {
      Vector3d a = Vector3d::Zero(), b = Vector3d::Zero();
      int idx_l = frame_count_l - it.start_frame;
      int idx_r = frame_count_r - it.start_frame;

      a = it.feature_per_frame[idx_l].point;

      b = it.feature_per_frame[idx_r].point;

      corres.push_back(make_pair(a, b));
      if (fea_ids != nullptr) {
        (*fea_ids).push_back(it.feature_id);
      }
    }
  }
  return corres;
}

VectorXd FeatureManager::getDepthVector() const {
  VectorXd dep_vec(getFeatureCount());
  int feature_index = -1;
  for (auto &it_per_id : feature) {
    if (!isLandmarkCandidate(it_per_id)) continue;

    dep_vec(++feature_index) = 1. / it_per_id.estimated_depth;
  }
  return dep_vec;
}

void FeatureManager::setDepth(const VectorXd &x) {
  int feature_index = -1;
  for (auto &it_per_id : feature) {
    if (!isLandmarkCandidate(it_per_id)) continue;

    it_per_id.estimated_depth = 1.0 / x(++feature_index);
    if (it_per_id.estimated_depth < 0) {
      it_per_id.solve_flag = SolveFail;
    } else
      it_per_id.solve_flag = SovelSucc;

    // LOG(INFO) << "feature id " << it_per_id->feature_id << " , start_frame "
    //           << it_per_id->start_frame << ", depth "
    //           << it_per_id->estimated_depth;
  }
}

void FeatureManager::removeFailures() {
  for (auto it = feature.begin(), it_next = feature.begin();
       it != feature.end(); it = it_next) {
    it_next++;
    if (it->solve_flag == SolveFail)  // FeatureManager::setDepth depth is less than zero
      feature.erase(it);
  }
}

void FeatureManager::clearDepth(const VectorXd &x) {
  int feature_index = -1;
  for (auto &it_per_id : feature) {
    if (!isLandmarkCandidate(it_per_id)) continue;

    it_per_id.estimated_depth = 1.0 / x(++feature_index);
  }
}

double FeatureManager::triangulateOneFeature(
    const FeaturePerId &feature_checked, const Eigen::Vector3d Ps_cam[],
    const Eigen::Matrix3d Rs_cam[]) const {
  int idx_i = feature_checked.start_frame;

  Eigen::MatrixXd svd_A(2 * feature_checked.feature_per_frame.size(), 4);
  int svd_idx = 0;

  Eigen::Matrix<double, 3, 4> P0;
  Eigen::Vector3d t0 = Ps_cam[idx_i];
  Eigen::Matrix3d R0 = Rs_cam[idx_i];
  P0.leftCols<3>() = Eigen::Matrix3d::Identity();
  P0.rightCols<1>() = Eigen::Vector3d::Zero();

  int idx_j = idx_i - 1;
  for (auto &it_per_frame : feature_checked.feature_per_frame) {
    idx_j++;

    Eigen::Vector3d t1 = Ps_cam[idx_j];
    Eigen::Matrix3d R1 = Rs_cam[idx_j];
    Eigen::Vector3d t = R0.transpose() * (t1 - t0);
    Eigen::Matrix3d R = R0.transpose() * R1;
    Eigen::Matrix<double, 3, 4> P;  // R0 to R1
    P.leftCols<3>() = R.transpose();
    P.rightCols<1>() = -R.transpose() * t;
    Eigen::Vector3d f = it_per_frame.point.normalized();  // f[2] != 1
    // https://www.uio.no/studier/emner/matnat/its/nedlagte-emner/UNIK4690/v16/forelesninger/lecture_7_2-triangulation.pdf
    // P = [P0 P1 P2]^T
    // A(2*i)   = x(i) * P3 - z(i) * P1
    // A(2*i+1) = y(i) * P3 - z(i) * P2
    svd_A.row(svd_idx++) = f[0] * P.row(2) - f[2] * P.row(0);
    svd_A.row(svd_idx++) = f[1] * P.row(2) - f[2] * P.row(1);
  }

  // depth*[u,v,1] = {}^C_WT * {}^WX,==> {}^WX = [x,y,z,1]
  ROS_ASSERT(svd_idx == svd_A.rows());
  Eigen::Vector4d svd_V =
      Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV)
          .matrixV()
          .rightCols<1>();
  double estimated_depth = svd_V[2] / svd_V[3];

  return estimated_depth;
}

void FeatureManager::triangulate(Vector3d Ps[], Matrix3d Rs[],
                                 double init_depth) {
  ROS_ASSERT(NUM_OF_CAM == 1);

  if (init_depth < 0) init_depth = INIT_DEPTH;

  for (auto &it_per_id : feature) {
    // if the feature has been observed many times and depth is large than zero
    if (!isLandmarkCandidate(it_per_id)) continue;

    if (it_per_id.estimated_depth > 0) continue;

    double depth = triangulateOneFeature(it_per_id, Ps, Rs);

    if (depth < 0.1) {
      it_per_id.estimated_depth = init_depth;
    } else {
      it_per_id.estimated_depth = depth;
    }
  }
}

void FeatureManager::removeBackShiftDepth(Eigen::Matrix3d marg_R,
                                          Eigen::Vector3d marg_P,
                                          Eigen::Matrix3d new_R,
                                          Eigen::Vector3d new_P) {
  for (auto it = feature.begin(), it_next = feature.begin();
       it != feature.end(); it = it_next) {
    it_next++;

    if (it->start_frame != 0)
      it->start_frame--;
    else {
      Eigen::Vector3d uv_i = it->feature_per_frame[0].point;
      // remove the oldest observation
      it->feature_per_frame.erase(it->feature_per_frame.begin());
      if (it->feature_per_frame.size() < 2) {
        feature.erase(it);
        continue;
      } else {
        // double depth = it->feature_per_frame.front().depth_from_lidar;
        // if (depth > 0) {
        //   it->estimated_depth = depth;
        //   continue;
        // }
        Eigen::Vector3d pts_i = uv_i * it->estimated_depth;
        Eigen::Vector3d w_pts_i = marg_R * pts_i + marg_P;
        Eigen::Vector3d pts_j = new_R.transpose() * (w_pts_i - new_P);
        double dep_j = pts_j(2);
        if (dep_j > 0)
          it->estimated_depth = dep_j;
        else
          // it->estimated_depth = INIT_DEPTH;
          it->estimated_depth = -1;
      }
    }
  }
}

void FeatureManager::removeBack() {
  for (auto it = feature.begin(), it_next = feature.begin();
       it != feature.end(); it = it_next) {
    it_next++;

    if (it->start_frame != 0)
      it->start_frame--;
    else {
      it->feature_per_frame.erase(it->feature_per_frame.begin());
      if (it->feature_per_frame.size() == 0) feature.erase(it);
    }
  }
}

void FeatureManager::removeFront(int frame_count) {
  for (auto it = feature.begin(), it_next = feature.begin();
       it != feature.end(); it = it_next) {
    it_next++;

    if (it->start_frame == frame_count) {
      it->start_frame--;
    } else {
      int j = WINDOW_SIZE - 1 - it->start_frame;
      if (it->endFrame() < frame_count - 1) continue;
      it->feature_per_frame.erase(it->feature_per_frame.begin() + j);
      if (it->feature_per_frame.size() == 0) feature.erase(it);
    }
  }
}

double FeatureManager::compensatedParallax2(const FeaturePerId &it_per_id,
                                            int frame_count) {
  // check the second last frame is keyframe or not
  // parallax between secondary last frame and third last frame
  const FeaturePerFrame &frame_i =
      it_per_id.feature_per_frame[frame_count - 2 - it_per_id.start_frame];
  const FeaturePerFrame &frame_j =
      it_per_id.feature_per_frame[frame_count - 1 - it_per_id.start_frame];

  double ans = 0;
  Vector3d p_j = frame_j.point;

  double u_j = p_j(0);
  double v_j = p_j(1);

  Vector3d p_i = frame_i.point;
  Vector3d p_i_comp;

  double dep_i = p_i(2);
  double u_i = p_i(0) / dep_i;
  double v_i = p_i(1) / dep_i;
  double du = u_i - u_j, dv = v_i - v_j;

  // int r_i = frame_count - 2;
  // int r_j = frame_count - 1;
  // p_i_comp = ric[camera_id_j].transpose() * Rs[r_j].transpose() * Rs[r_i] *
  // ric[camera_id_i] * p_i;
  p_i_comp = p_i;
  double dep_i_comp = p_i_comp(2);
  double u_i_comp = p_i_comp(0) / dep_i_comp;
  double v_i_comp = p_i_comp(1) / dep_i_comp;
  double du_comp = u_i_comp - u_j, dv_comp = v_i_comp - v_j;

  ans = max(
      ans, sqrt(min(du * du + dv * dv, du_comp * du_comp + dv_comp * dv_comp)));

  return ans;
}
}  // namespace clic