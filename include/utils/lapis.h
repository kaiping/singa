// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-14 18:48
#ifndef INCLUDE_UTILS_LAPIS_H_
#define INCLUDE_UTILS_LAPIS_H_
#include <memory>
#include <random>

#include "Eigen/Core"
#include "Eigen/Dense"

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
        Eigen::RowMajor> EigenMatrix;
typedef Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic,
        Eigen::RowMajor> EigenArray;
typedef Eigen::Array<float, 1,  Eigen::Dynamic, Eigen::RowMajor> EigenAVector;
typedef Eigen::Matrix<float, 1,  Eigen::Dynamic, Eigen::RowMajor> EigenMVector;

typedef Eigen::Map<EigenMatrix> MMat;
typedef Eigen::Map<EigenMVector> MVec;
typedef Eigen::Map<EigenArray> AMat;
typedef Eigen::Map<EigenAVector> AVec;

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
      instance_.reset(new Lapis());
    }
    return instance_;
  }

  std::shared_ptr<std::mt19937>& generator() {
    return generator_;
  }

 private:
  Lapis();
  std::shared_ptr<std::mt19937> generator_;
  static std::shared_ptr<Lapis> instance_;
};
}  // namespace lapis
#endif  // INCLUDE_UTILS_LAPIS_H_
