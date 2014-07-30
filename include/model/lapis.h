// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-14 18:48
#ifndef INCLUDE_UTILS_LAPIS_H_
#define INCLUDE_UTILS_LAPIS_H_
#include <memory>
#include <chrono>
#include "mshadow/tensor.h"
#include "mshadow/tensor_container.h"

namespace lapis {
typedef mshadow::TensorContainer<mshadow::cpu, 4> Blob4;
typedef mshadow::TensorContainer<mshadow::cpu, 3> Blob3;
typedef mshadow::TensorContainer<mshadow::cpu, 2> Blob2;
typedef mshadow::TensorContainer<mshadow::cpu, 1> Blob1;
typedef mshadow::Tensor<mshadow::cpu, 4> Tensor4;
typedef mshadow::Tensor<mshadow::cpu, 3> Tensor3;
typedef mshadow::Tensor<mshadow::cpu, 2> Tensor2;
typedef mshadow::Tensor<mshadow::cpu, 1> Tensor1;


typedef mshadow::Shape<4> TShape4;
typedef mshadow::Shape<3> TShape3;
typedef mshadow::Shape<2> TShape2;
typedef mshadow::Shape<1> TShape1;
typedef mshadow::Random<mshadow::cpu> Random;
auto Shape1=mshadow::Shape1;
auto Shape2=mshadow::Shape2;
auto Shape3=mshadow::Shape3;
auto Shape4=mshadow::Shape4;

using mshadow::expr::reshape;
/**
 * Class Lapis provide some global object, e.g., random number generator
 */
class Lapis {
 public:
  /**
   * Singleton of the class
   */
  inline static std::shared_ptr<Lapis> &Instance() {
    if (!instance_.get()) {
      unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
      Random rnd(seed);
      instance_.reset(new Lapis(rnd));
    }
    return instance_;
  }

  mshadow::Random<mshadow::cpu> &rnd() {
    return rnd_;
  }

 private:
  Lapis(Random &rnd): rnd_(rnd) {}
  Random rnd_;
  static std::shared_ptr<Lapis> instance_;
};
}  // namespace lapis
#endif  // INCLUDE_UTILS_LAPIS_H_
