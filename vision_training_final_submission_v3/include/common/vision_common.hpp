#pragma once

#include <algorithm>
#include <filesystem>
#include <string>

namespace vision_common {

inline void ensureDirectory(const std::string& path) {
    std::filesystem::create_directories(path);
}

template <typename T>
inline T clampValue(T value, T low, T high) {
    return std::max(low, std::min(value, high));
}

} // namespace vision_common
