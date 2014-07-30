// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-14 18:48
#ifndef INCLUDE_UTILS_LAPIS_H_
#define INCLUDE_UTILS_LAPIS_H_
#include <memory>
#include <chrono>

typedef mshadow::Tensor<cpu, 4> Blob4;
typedef mshadow::Tensor<cpu, 3> Blob3;
typedef mshadow::Tensor<cpu, 2> Blob2;
typedef mshadow::Tensor<cpu, 1> Blob1;
typedef mshadow::Shape<4> Blob4;
typedef mshadow::Shape<3> Blob3;
typedef mshadow::Shape<2> Blob2;
typedef mshadow::Shape<1> Blob1;
typedef mshadow::Random<cpu> Random;
namespace lapis {
/**
 * Class Lapis provide some global object, e.g., random number generator
 */
class Lapis {
 public:
  /**
   * Singleton of the class
   */
  inline static std::shared_ptr<Lapis>& Instance() {
    if (!instance_.get()) {
      unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
      Random rnd(seed);
      instance_.reset(new Lapis(rnd));
    }
    return instance_;
  }

  mshadow::Random<cpu>& rnd() {
    return rnd_;
  }

 private:
  Lapis(Random& rnd):rnd_(rnd){}
  Random rnd_;
  static std::shared_ptr<Lapis> instance_;
};
}  // namespace lapis
#endif  // INCLUDE_UTILS_LAPIS_H_
