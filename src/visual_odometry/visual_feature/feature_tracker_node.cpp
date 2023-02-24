
#include "feature_tracker_node.h"

#define SHOW_UNDISTORTION 0

using namespace std;

namespace feature_tracker {

FeatureTrackerNode::FeatureTrackerNode(std::string &config_file,
                                       bool offline_mode,
                                       double add_extra_timeoffset_s)
    : first_image_time(0),
      last_image_time(0),
      first_image_flag(true),
      init_pub(false),
      pub_count(1),
      offline_mode_(offline_mode),
      add_extra_timeoffset_s_(add_extra_timeoffset_s) {
  LOG(INFO) << "[FeatureTrackerNode] load  " << config_file;
  readParameters(config_file);

  for (int i = 0; i < NUM_OF_CAM; i++)
    trackerData[i].readIntrinsicParameter(CAM_NAMES[i]);

  if (FISHEYE) {
    for (int i = 0; i < NUM_OF_CAM; i++) {
      trackerData[i].fisheye_mask = cv::imread(FISHEYE_MASK, 0);
      if (!trackerData[i].fisheye_mask.data) {
        LOG(INFO) << "load mask fail";
        ROS_BREAK();
      } else
        LOG(INFO) << "load mask success";
    }
  }

  LOG(INFO) << "FeatureTrackerNode subscribe image topic " << IMAGE_TOPIC;

  ros::NodeHandle nh;
  if (!offline_mode_) {
    sub_img = nh.subscribe(IMAGE_TOPIC, 0, &FeatureTrackerNode::img_callback,
                           this, ros::TransportHints().tcpNoDelay());
  }

  pub_img = nh.advertise<sensor_msgs::PointCloud>("image_feature", 1000);
  pub_match = nh.advertise<sensor_msgs::Image>("feature_img", 1000);
  pub_restart = nh.advertise<std_msgs::Bool>("restart", 1000);
}

void FeatureTrackerNode::img_callback(
    const sensor_msgs::ImageConstPtr &img_msg) {
  std_msgs::Header img_header = img_msg->header;
  // add_extra_timeoffset s
  img_header.stamp += ros::Duration(add_extra_timeoffset_s_);

  double t_img_cur = img_header.stamp.toSec();
  if (first_image_flag) {
    first_image_flag = false;
    first_image_time = t_img_cur;
    last_image_time = t_img_cur;
    return;
  }
  // detect unstable camera stream
  if (t_img_cur - last_image_time > 1.0 || t_img_cur < last_image_time) {
    LOG(INFO) << "image discontinue! reset the feature tracker!";
    first_image_flag = true;
    last_image_time = 0;
    pub_count = 1;
    std_msgs::Bool restart_flag;
    restart_flag.data = true;
    pub_restart.publish(restart_flag);
    return;
  }
  last_image_time = t_img_cur;
  // frequency control
  double cur_freq = 1.0 * pub_count / (t_img_cur - first_image_time);
  if (round(cur_freq) <= LK_DESIRED_FREQ) {
    PUB_THIS_FRAME = true;
    // reset the frequency control
    if (abs(cur_freq - LK_DESIRED_FREQ) < 0.01 * LK_DESIRED_FREQ) {
      first_image_time = t_img_cur;
      pub_count = 0;
    }
  } else
    PUB_THIS_FRAME = false;

  if (img_msg->encoding == "8UC1" || img_msg->encoding == "mono8") {
    sensor_msgs::Image img;
    img.header = img_header;
    img.height = img_msg->height;
    img.width = img_msg->width;
    img.is_bigendian = img_msg->is_bigendian;
    img.step = img_msg->step;
    img.data = img_msg->data;
    img.encoding = "mono8";
    ptr_ = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
  } else if (img_msg->encoding == "8UC3") {
    sensor_msgs::Image img;
    img.header = img_header;
    img.height = img_msg->height;
    img.width = img_msg->width;
    img.is_bigendian = img_msg->is_bigendian;
    img.step = img_msg->step;
    img.data = img_msg->data;
    img.encoding = "bgr8";
    ptr_ = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::BGR8);
  } else {
    ptr_ = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);
  }

  cv::Mat show_img = ptr_->image;
  TicToc t_r;
  for (int i = 0; i < NUM_OF_CAM; i++) {
    ROS_DEBUG("processing camera %d", i);
    if (i != 1 || !STEREO_TRACK) {
      if (show_img.channels() != 1) {
        cv::cvtColor(show_img, show_img, CV_RGB2GRAY);
        trackerData[i].readImage(show_img, t_img_cur);
      } else {
        trackerData[i].readImage(ptr_->image.rowRange(ROW * i, ROW * (i + 1)),
                                 t_img_cur);
      }

    } else {
      if (EQUALIZE) {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
        clahe->apply(ptr_->image.rowRange(ROW * i, ROW * (i + 1)),
                     trackerData[i].cur_img);
      } else
        trackerData[i].cur_img = ptr_->image.rowRange(ROW * i, ROW * (i + 1));
    }

#if SHOW_UNDISTORTION
    trackerData[i].showUndistortion("undistrotion_" + std::to_string(i));
#endif
  }

  for (unsigned int i = 0;; i++) {
    bool completed = false;
    for (int j = 0; j < NUM_OF_CAM; j++)
      if (j != 1 || !STEREO_TRACK) completed |= trackerData[j].updateID(i);
    if (!completed) break;
  }

  if (PUB_THIS_FRAME) {
    pub_count++;
    sensor_msgs::PointCloudPtr feature_points(new sensor_msgs::PointCloud);
    sensor_msgs::ChannelFloat32 id_of_point;
    sensor_msgs::ChannelFloat32 u_of_point;
    sensor_msgs::ChannelFloat32 v_of_point;
    sensor_msgs::ChannelFloat32 velocity_x_of_point;
    sensor_msgs::ChannelFloat32 velocity_y_of_point;

    feature_points->header = img_header;
    feature_points->header.frame_id = "world";

    vector<set<int>> hash_ids(NUM_OF_CAM);
    for (int i = 0; i < NUM_OF_CAM; i++) {
      auto &un_pts = trackerData[i].cur_un_pts;
      auto &cur_pts = trackerData[i].cur_pts;
      auto &ids = trackerData[i].ids;
      auto &pts_velocity = trackerData[i].pts_velocity;
      for (unsigned int j = 0; j < ids.size(); j++) {
        if (trackerData[i].track_cnt[j] > 1) {
          int p_id = ids[j];
          hash_ids[i].insert(p_id);
          geometry_msgs::Point32 p;
          p.x = un_pts[j].x;
          p.y = un_pts[j].y;
          p.z = 1;

          feature_points->points.push_back(p);
          id_of_point.values.push_back(p_id * NUM_OF_CAM + i);
          u_of_point.values.push_back(cur_pts[j].x);
          v_of_point.values.push_back(cur_pts[j].y);
          velocity_x_of_point.values.push_back(pts_velocity[j].x);
          velocity_y_of_point.values.push_back(pts_velocity[j].y);
        }
      }
    }
    feature_points->channels.push_back(id_of_point);
    feature_points->channels.push_back(u_of_point);
    feature_points->channels.push_back(v_of_point);
    feature_points->channels.push_back(velocity_x_of_point);
    feature_points->channels.push_back(velocity_y_of_point);
    ROS_DEBUG("publish %f, at %f", feature_points->header.stamp.toSec(),
              ros::Time::now().toSec());
    // skip the first image; since no optical speed on frist image
    if (!init_pub) {
      init_pub = true;
    } else {
      // if (offline_mode_) {
      image_feature_buf_.push_back(feature_points);
      //} else {
      //  pub_img.publish(feature_points);
      // }
    }

    if (SHOW_TRACK) {
      ptr_ = cv_bridge::cvtColor(ptr_, sensor_msgs::image_encodings::BGR8);
      // cv::Mat stereo_img(ROW * NUM_OF_CAM, COL, CV_8UC3);
      cv::Mat stereo_img = ptr_->image;

      for (int i = 0; i < NUM_OF_CAM; i++) {
        cv::Mat tmp_img = stereo_img.rowRange(i * ROW, (i + 1) * ROW);
        cv::cvtColor(show_img, tmp_img, CV_GRAY2RGB);

        for (unsigned int j = 0; j < trackerData[i].cur_pts.size(); j++) {
          double len = std::min(1.0, 1.0 * trackerData[i].track_cnt[j] / 20);
          cv::circle(tmp_img, trackerData[i].cur_pts[j], 2,
                     cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
#if false
          // draw speed line
          Vector2d tmp_cur_un_pts(trackerData[i].cur_un_pts[j].x,
                                  trackerData[i].cur_un_pts[j].y);
          Vector2d tmp_pts_velocity(trackerData[i].pts_velocity[j].x,
                                    trackerData[i].pts_velocity[j].y);
          Vector3d tmp_prev_un_pts;
          tmp_prev_un_pts.head(2) = tmp_cur_un_pts - 0.10 * tmp_pts_velocity;
          tmp_prev_un_pts.z() = 1;
          Vector2d tmp_prev_uv;
          trackerData[i].m_camera->spaceToPlane(tmp_prev_un_pts, tmp_prev_uv);
          cv::line(tmp_img, trackerData[i].cur_pts[j],
                   cv::Point2f(tmp_prev_uv.x(), tmp_prev_uv.y()),
                   cv::Scalar(255, 0, 0), 1, 8, 0);

          char name[10];
          sprintf(name, "%d", trackerData[i].ids[j]);
          cv::putText(tmp_img, name, trackerData[i].cur_pts[j],
                      cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
#endif
        }
      }
      // cv::imshow("vis", stereo_img);
      // cv::waitKey(5);
      pub_match.publish(ptr_->toImageMsg());
    }
  }
  LOG(INFO) << "whole feature tracker processing costs: " << t_r.toc() << " ms";
}

template <typename T>
T readParam(ros::NodeHandle &n, std::string name) {
  T ans;
  if (n.getParam(name, ans)) {
    ROS_INFO_STREAM("Loaded " << name << ": " << ans);
  } else {
    ROS_ERROR_STREAM("Failed to load " << name);
    n.shutdown();
  }
  return ans;
}
}  // namespace feature_tracker

int main(int argc, char **argv) {
  ros::init(argc, argv, "feature_tracker_node");
  ros::NodeHandle nh("~");
  ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME,
                                 ros::console::levels::Info);

  std::string config_file;
  config_file = feature_tracker::readParam<std::string>(nh, "config_file");

  feature_tracker::FeatureTrackerNode feature_tracker_node(config_file, false);
  /*
  if (SHOW_TRACK)
      cv::namedWindow("vis", cv::WINDOW_NORMAL);
  */
  ros::spin();
  return 0;
}

// new points velocity is 0, pub or not?
// track cnt > 1 pub?
