// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-14 18:48
#ifndef INCLUDE_MODEL_LAPIS_H_
#define INCLUDE_MODEL_LAPIS_H_
#include <glog/logging.h>
#include <memory>
#include <chrono>
#include <sstream>

#include "mshadow/tensor.h"

namespace lapis {
typedef mshadow::Tensor<mshadow::cpu, 4> Tensor4;
typedef mshadow::Tensor<mshadow::cpu, 3> Tensor3;
typedef mshadow::Tensor<mshadow::cpu, 2> Tensor2;
typedef mshadow::Tensor<mshadow::cpu, 1> Tensor1;
typedef mshadow::Random<mshadow::cpu> Random;
using mshadow::Shape1;
using mshadow::Shape2;
using mshadow::Shape3;
using mshadow::Shape4;
/**
 * Blob stores the data of both parameters and (raw/intermediate) features.
 */
class Blob {
 public:
  Blob(): dptr(nullptr), num_(0), channels_(0),
          height_(0), width_(0), length_(0){}
  Blob(int num, int channels, int height, int width);
  ~Blob() {
    VLOG(3)<<"Free Blob at "<<dptr;
    if(dptr!=nullptr)
      delete dptr;
  }
  const std::string tostring() {
    std::stringstream ss;
    ss<<"("<<num_<<" "<<channels_<<" "<<height_<<" "<<width_<<")";
    return ss.str();
  }
  /**
   * allocate memory of size length, e.g., for the bias parameter
   * do nothing if it is of exactly the same shape
   * @param length e.g., length of the bias parameter
  void Resize(int length);
   */
  /**
   * allocate memory of size height*width, e.g., for the weight matrix
   * do nothing if it is of exactly the same shape
   * @param height e.g., num of rows of the weight matrix
   * @param width e.g., num of cols of the weight matrix
  void Resize(int width, int height);
   */
  /**
   * allocate memory of size num*height*width, e.g., for the gray image
   * do nothing if it is of exactly the same shape
   * @param num number of instances (image) per blob
   * @param height e.g., height of the image
   * @param width e.g., width of the image
  void Resize(int width, int height, int num);
   */
  /**
   * allocate memory of size num*height*width, e.g., for the gray image
   * do nothing if it is of exactly the same shape
   * @param num number of instances (image) per blob
   * @param height e.g., num of rows of the weight matrix
   * @param width e.g., num of cols of the weight matrix
   */
  void Resize(int num, int channels, int height, int width);

  /**
   * Return num of instances stored in this blob
   */
  const int num() const {
    return num_;
  }
  /**
   * Return channels of this blob
   */
  const int channels() const {
    return channels_;
  }
  /**
   * For image data, it is the height of the image;
   * For matrix parameters, it is the num of rows in the matrix;
   */
  const int height() const {
    return height_;
  }
  /**
   * For image data, it is the width of the image;
   * For matrix parameters, it is the num of cols in the matrix;
   */
  const int width() const {
    return width_;
  }

  /**
   * Return the total size in terms of floats
   */
  const int length() const {
    return length_;
  }

  /**
   * Return the size of one record in terms of floats
   */
  const int record_length() const {
    return record_length_;
  }

  /**
   * return the mem size of all Blobs in terms of bytes
   */
  static const int MSize() {
    return count_*sizeof(float)/(1024*1024);
  }

  int Gt(float v);
  int Lt(float v);
  bool Nan();
  float *dptr;
 private:
  int num_, channels_, height_, width_;
  unsigned int length_, record_length_;
  static int count_;
};


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

  Random &rnd() {
    return rnd_;
  }

 private:
  Lapis(Random &rnd): rnd_(rnd) {}
  Random rnd_;
  static std::shared_ptr<Lapis> instance_;
};
}  // namespace lapis
#endif  // INCLUDE_MODEL_LAPIS_H_
