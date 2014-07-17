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

  const float *Content() const {
    return content_;
  }
  float *MutableContent() const {
    return content_;
  }
  const int Height() const {
    return height_;
  }
  const int Width() const {
    return width_;
  }
  const int Size() const {
    return size_;
  }

 private:
  int size_, num_, channels_, height_, width_;
  float *content_;
};

}  // namespace lapis
#endif  // INCLUDE_MODEL_BLOB_H_
