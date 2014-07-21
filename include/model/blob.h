// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-06 14:13

#ifndef INCLUDE_MODEL_BLOB_H_
#define INCLUDE_MODEL_BLOB_H_

#include <vector>
#include <string>

#include "proto/lapis.pb.h"

namespace lapis {
/**
 * Blob stores the content of both parameters and (raw/intermediate) features.
 */
class Blob {
 public:
  Blob(): size_(0), channels_(0), height_(0),
    width_(0), content_(nullptr) {}
  /**
   * allocate memory of size length, e.g., for the bias parameter
   * do nothing if it is of exactly the same shape
   * @param length e.g., length of the bias parameter
   */
  void Reshape(int length);
  /**
   * allocate memory of size height*width, e.g., for the weight matrix
   * do nothing if it is of exactly the same shape
   * @param height e.g., num of rows of the weight matrix
   * @param width e.g., num of cols of the weight matrix
   */
  void Reshape(int height, int width);
  /**
   * allocate memory of size num*height*width, e.g., for the gray image
   * do nothing if it is of exactly the same shape
   * @param num number of instances (image) per blob
   * @param height e.g., height of the image
   * @param width e.g., width of the image
   */
  void Reshape(int num, int height, int width);
  /**
   * allocate memory of size num*height*width, e.g., for the gray image
   * do nothing if it is of exactly the same shape
   * @param num number of instances (image) per blob
   * @param height e.g., num of rows of the weight matrix
   * @param width e.g., num of cols of the weight matrix
   */
  void Reshape(int num, int channel, int height, int width);

  const float *content() const {
    return content_;
  }
  float *mutable_content() const {
    return content_;
  }
  /**
   * Return num of instances stored in this blob
   */
  const int size() const {
    return size_;
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
  const int Length() const {
    return size_ * channels_ * height_ * width_;
  }

 private:
  int size_, channels_, height_, width_;
  float *content_;
};

}  // namespace lapis
#endif  // INCLUDE_MODEL_BLOB_H_
