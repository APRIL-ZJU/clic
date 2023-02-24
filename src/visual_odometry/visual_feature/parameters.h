#pragma once
#include <ros/ros.h>

namespace feature_tracker {

extern int ROW;            // image width
extern int COL;            // image height
extern int FOCAL_LENGTH;   // focal
const int NUM_OF_CAM = 1;  // 

///===== feature_tracker_node parameter ==== ///
extern std::string IMAGE_TOPIC;             // 
extern std::string FISHEYE_MASK;            // file path of fisheye mask
extern std::vector<std::string> CAM_NAMES;  // filename of camera config

extern int LK_DESIRED_FREQ;  // frequency control of LK tracking
extern int SHOW_TRACK;       // if we publish the tracking images
extern int STEREO_TRACK;     // if we use stereo

// also in FeatureTracker
extern bool PUB_THIS_FRAME;  // 

///===== FeatureTracker parameter ==== ///
extern double F_THRESHOLD;  // for findFundamentalMat in rejectWithF
extern int MAX_CNT;         // max number of tracking
extern int MIN_DIST;  // min distance for goodFeaturesToTrack
extern int EQUALIZE;  //  
extern int FISHEYE;  // 

extern int FLOW_BACK;        //  
extern double FB_THRESHOLD;  //  

void readParameters(std::string& config_file);

}  // namespace feature_tracker
