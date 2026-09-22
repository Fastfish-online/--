#pragma once

namespace task3_config {
constexpr int hsv_h_low = 0;
constexpr int hsv_h_high = 40;
constexpr int hsv_s_low = 45;
constexpr int hsv_v_low = 35;
constexpr int hough_min_radius = 12;
constexpr int hough_max_radius = 38;
constexpr int hough_min_distance = 22;
constexpr double hough_dp = 1.0;
constexpr double hough_param1 = 70.0;
constexpr double hough_param2 = 12.0;
constexpr int lost_tolerance_frames = 60;
constexpr float mechanism_radius_min = 55.0f;
constexpr float mechanism_radius_max = 135.0f;
constexpr float circle_inlier_tolerance = 12.0f;
} // namespace task3_config
