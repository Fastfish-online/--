#pragma once

namespace task2_config {
// omega(t) = b + A * sin(Omega * t + phi)
constexpr double known_radius_px = 220.0;
constexpr double center_x_px = 480.0;
constexpr double center_y_px = 360.0;
constexpr double amplitude_lower_bound = 0.0;
constexpr double omega_lower_bound = 0.0;
constexpr double mean_rate_must_exceed_amplitude = 1.0;
} // namespace task2_config
