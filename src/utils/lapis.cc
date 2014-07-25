// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-22 19:53
#include <chrono>
#include "utils/lapis.h"
namespace lapis {
std::shared_ptr<Lapis> Lapis::instance_;
Lapis::Lapis() {
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  generator_=std::make_shared<std::mt19937>(seed);
}
}  // namespace lapis

