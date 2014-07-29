// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-06 14:13

#ifndef INCLUDE_MODEL_TENSOR_H_
#define INCLUDE_MODEL_TENSOR_H_

#include <vector>
#include <cstring>
#include <memory>


#include "proto/model.pb.h"
using std::shared_ptr;
typedef shared_ptr<Tensor> TensorPtr;
namespace lapis {
/**
 * Tensor with at most 4 dimensions.
 * It stores the data of both parameters and (raw/intermediate) features, and
 * provides operations, e.g., addition, substraction, multiplication (e.g., dot
 * or element-wise multiplication), etc.
 */
class Tensor {
 public:
  Blob(): num_(0), channels_(0), height_(0), width_(0), data_(nullptr) {}
  Blob(int num, int channels, int height, int width);
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

  void Reset(int n, int c, int h, int w) ;

  float* at(int idx0, int idx1=0, int idx2=0, idx3=0) {
    CHECK(idx1==0 || dim_>1);
    CHECK(idx2==0 || dim_>2);
    CHECK(idx3==0 || dim_>3);
    return data_+ (idx0*len[1]+idx1*len[2]+idx2*len[3]+idx3);
  }

  int shape(int d) {
    CHECK(d<dim_);
    return shape_[d];
  }
  TensorPtr Slice(int start);
  int offset(int n, int c=0, int h=0 ,int w=0) const {
    return ((n*num_+c)*height_+h)*width_+w;
  }

  const float *data() const {
    return data_;
  }

  void SetOne();
  void SetZero();
  /**
   * C=A*B.
   */
  void Dot(TensorPtr A, TensorPtr B, TensorPtr C,
      bool transA=false, bool transB=false, bool overwrite=true);
  /**
   * A is matrix, B is vector, add B to each column of A
   */
  void AddColumn(Tensor&& B,Tensor* A);

  void Copy(Tensor&& src,Tensor* dest);
  void Add(Tensor&& t, Tensor*dest);
  void Add(Tensor&& t, Tensor*dest);
  void MultScalar(float factor, Tensor& src, Tensor* dest);
  void Sum(Tensor&& src, int axis, Tensor&& dest);
  void Sum(Tensor&& src, int axis, Tensor* dest);
  void Square(Tensor&& src, Tensor&&dest);
  void Mult(const Tensor& src, Tensor* dest);
 private:
  //! totoal num of dimensions
  int dim_;
  //! length of each dimension
  int shape[4];
  //! len[0] total length, len[i] len[0]/(len[i-1]*shape[i-1])
  int len[4];
  float* data_;
};

}  // namespace lapis
#endif  // INCLUDE_MODEL_TENSOR_H_
