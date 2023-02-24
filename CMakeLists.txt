cmake_minimum_required(VERSION 3.0.2)
project(clic)

set(CMAKE_BUILD_TYPE "Release")
#set(CMAKE_CXX_FLAGS "-std=c++17")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g -pthread")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -msse4.2")

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()


find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  sensor_msgs
  geometry_msgs
  nav_msgs
  visualization_msgs
  eigen_conversions
  pcl_conversions
  pcl_ros
  cv_bridge
  roslib
  rosbag
  tf
  message_generation
)

find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Ceres REQUIRED) 
# find_package(Sophus REQUIRED) # sophus has been included in clic/src/sophus_lib
find_package(OpenCV REQUIRED)

# find_package(PkgConfig REQUIRED)
# pkg_check_modules(YAML_CPP REQUIRED yaml-cpp>=0.5)

## Generate messages in the 'msg' folder
add_message_files(
  FILES
  feature_cloud.msg
  imu_array.msg
  pose_array.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs 
  sensor_msgs
)


###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
  INCLUDE_DIRS src
#  LIBRARIES clic
  CATKIN_DEPENDS std_msgs
#  DEPENDS system_lib
)


## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  src
  ${catkin_INCLUDE_DIRS}
  ${YAML_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
  # ${Sophus_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

list(APPEND thirdparty_libraries
        ${YAML_CPP_LIBRARIES}
        ${Boost_LIBRARIES}
        ${catkin_LIBRARIES}
        ${PCL_LIBRARIES}
        ${OpenCV_LIBS}
        ${CERES_LIBRARIES}
)

######################
### Library
######################
# Continuous Time Trajectory
add_library(spline_lib src/spline/trajectory.cpp)
target_link_libraries(spline_lib ${thirdparty_libraries})

# LiDAR Odometry
file(GLOB lidar_odometry_files
    "src/lidar_odometry/*.cpp"
)
add_library(lidar_lib ${lidar_odometry_files})
target_link_libraries(lidar_lib spline_lib ${thirdparty_libraries})

file(GLOB visual_feature_files
    "src/visual_odometry/visual_feature/*.cpp"
    "src/visual_odometry/visual_feature/camera_models/*.cc"
)
add_library(feature_tracker_lib ${visual_feature_files})
target_link_libraries(feature_tracker_lib ${thirdparty_libraries})

# Visual Odometry
file(GLOB visual_odometry_files
    "src/visual_odometry/*.cpp"
    "src/visual_odometry/initial/*.cpp"
)
add_library(visual_lib ${visual_odometry_files})
target_link_libraries(visual_lib spline_lib ${thirdparty_libraries})

######################
### Nodes
######################
# Visual Feature Tracker
add_executable(feature_tracker ${visual_feature_files})
target_link_libraries(feature_tracker ${thirdparty_libraries})

# Estimator
add_executable(odometry_node 
   src/app/odometry_node.cpp
   src/inertial/imu_state_estimator.cpp
   src/inertial/inertial_initializer.cpp
   src/estimator/msg_manager.cpp
   src/estimator/trajectory_manager.cpp
   src/estimator/trajectory_estimator.cpp
   src/estimator/odometry_manager.cpp
   src/estimator/factor/analytic_diff/marginalization_factor.cpp
   src/utils/parameter_struct.cpp
   src/loop_closure/loop_closure.cpp
   src/loop_closure/pose_graph.cpp 
)

target_link_libraries(odometry_node 
   spline_lib 
   lidar_lib 
   visual_lib  
   feature_tracker_lib
   ${thirdparty_libraries}
)

# About bag
add_executable(recovery_vlp16_timestamp src/app/recovery_vlp16_timestamp.cpp)
target_link_libraries(recovery_vlp16_timestamp lidar_lib ${thirdparty_libraries})

# Test

add_executable(test_analytic_factor 
  src/app/test_analytic_factor.cpp
  src/utils/parameter_struct.cpp
  src/visual_odometry/parameters.cpp)
  
target_link_libraries(test_analytic_factor spline_lib ${thirdparty_libraries})

