#pragma once

#include <utils/yaml_utils.h>
#include <iostream>

namespace clic {

const int WINDOW_SIZE = 10;
const int NUM_OF_CAM = 1;

// For Pointgrey camera. Euroc dataset is 460.0;
extern double FOCAL_LENGTH;

extern double INIT_DEPTH;
extern double MIN_PARALLAX;

extern double ACC_N, ACC_W;
extern double GYR_N, GYR_W;

void readParameters(const YAML::Node& node);

}  // namespace clic
