// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-14 18:48
#ifndef INCLUDE_UTILS_LAPIS_H_
#define INCLUDE_UTILS_LAPIS_H_
#include <memory>
#include <random>

#include "Eigen/Core"
#include "Eigen/Dense"

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
        Eigen::RowMajor> MatrixType;
typedef Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic,
        Eigen::RowMajor> ArrayType;
typedef Eigen::Map<MatrixType> MapMatrixType;
typedef Eigen::Map<ArrayType> MapArrayType;
typedef Eigen::Map<Eigen::RowVectorXf> MapVectorType;
namespace lapis {

class Lapis {
 public:
  inline static Lapis& Get() {
    if (!instance_.get()) {
      instance_.reset(new Lapis());
    }
    return *instance_;
  }

  void set_random_seed(const unsigned int seed) {
    rng_.seed(seed);
  }

  std::mt19937 &rng() {return rng_;}

 private:
  Lapis(){};
  std::mt19937 rng_;
  static std::shared_ptr<Lapis> instance_;
};
}  // namespace lapis
#endif  // INCLUDE_UTILS_LAPIS_H_
