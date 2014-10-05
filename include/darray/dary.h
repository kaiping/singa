// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-09-10 16:34

#ifndef INCLUDE_DARRAY_DARY_H_
#define INCLUDE_DARRAY_DARY_H_

#include <vector>
#include <utility>
#include <functional>
#include "proto/model.pb.h"


using std::vector;
namespace lapis {
using Range=std::pair<int, int>;
class Shape;
class DAry {
 public:
  DAry(){}
  /**
   * create a new dary with data and partition from other dary,
   * but a new shape, this new shape only diff with other dary on
   * the first or last few dimension, total size should the same;
   * like Reshape();
   */
  DAry(const DAry& other, const vector<int>& shape);

  /**
   * init with the same shape and partition as other, may copy data
   */
  DAry(const DAry& other, bool copy);
  /**
   * swap dptr
   */
  void SwapDptr(DAry* other);

  const float* dptr() const;
  float* mutable_dptr();

  /**
   * true if memory has allocated or false
   */
  bool allocated();

  /**
   * set shape and partition from proto
   */
  void InitFromProto(const DAryProto& proto);
  void ToProto(DAryProto* proto, bool copyData);
  /**
   * set shape and partition as other dary
   */
  void InitLike(const DAry& other);
  /**
   * set shape if no set before; otherwise check with old shape
   */
  void SetShape(const vector<int>& shape) ;
  void SetShape(const Shape& shape) ;

  void SampleGaussian(float mean, float std);
  void SampleUniform(float mean, float std);
  /**
   * allocate memory
   */
  void AllocateMemory();
  void FreeMemory();

  /**
   * subdary on the 0-th dim
   */
  DAry operator[](int k) const;
  DAry operator[](int k);

  /**
   * Dot production
   */
  void Dot( const DAry& src1, const DAry& src2, bool trans1=false, bool trans2=false);
  void Mult( const DAry& src1, const DAry& src2);
  void Mult( const DAry& src1, const float x);
  void Div( const DAry& src1, const float x);
  void Div( const DAry& src1, const DAry& x);

  void Square( const DAry& src);
  /**
   * sum along dim-th dimension, with range r
   * # of dimensions of dst is that of src-1
   */
  void Sum( const DAry& src, int dim, Range r);
  void Copy( const DAry& src);
  /**
   * dst=src1-src2
   */
  void Minus( const DAry& src1, const DAry& src2);
  /**
   * minus this=this-src
   */
  void Minus( const DAry& src);
  /**
   * dst=src1+src2
   */
  void Add( const DAry& src1, const DAry& src2);
  /**
   * this=this+src
   */
  void Add( const DAry& src);
  /**
   * dst=src1+x
   */
  void Add( const DAry& src1, const float x);
  /**
   * dst=src^x
   */
  void Pow( const DAry& src1, const float x);


  /**
   * Add the src to dst as a vector along dim-th dimension
   * i.e., the dim-th dimension should have the same length as src
   */
  void AddVec( const DAry& src, int dim);

  /**
   * src must be a matrix
   * this is a vector
   */
  void SumRows(const DAry& src);

  /**
   * sum the src except the dim-th dimension
   * e.g., let src be a tensor with shape (2,4,5), then SumExcept(dst, src,1)
   * would results a vector of length 4
   */
  void SumExcept(const DAry& src, int dim);

  /**
   * sum all elements
   */
  float Sum();


  /**
   * check whether the element at index is local
   */
  bool isLocal(vector<int> index);

  /**
   * return the local index range for dim-th dimension
   */
  Range IndexRange(int dim) const;

  void set(float x);
  /**
   * fetch data to local according to index ranges
   * create a new DAry which has the same shape as this dary, but
   * the requested data are local
   */
  DAry FetchData(const vector<Range>& slice) const;

  /**
   * return the ref for the ary at this index
   * check shape
   */
  float& at(int idx0,int idx1, int idx2, int idx3) const;
  float& at(int idx0,int idx1, int idx2) const;
  float& at(int idx0,int idx1) const;
  float& at(int idx0) const;

/**
   * return the value for the ary at this index
   * check shape
   */
  const float get(int idx0,int idx1, int idx2, int idx3) const;
  const float get(int idx0,int idx1, int idx2) const;
  const float get(int idx0,int idx1) const;
  const float get(int idx0) const;


  /**
   * put max(src,x) into dst
   */
  void Max( const DAry& src, float x);

  /**
   * max element
   */
  float Max();

  /**
   * apply the func for every element in src, put result to dst
   */
  void Map( std::function<float(float)> func, const DAry& src);
  void Map( std::function<float(float, float)> func, const DAry& src1, const DAry& src2);
  void Map( std::function<float(float, float, float)> func, const DAry& src1, const DAry& src2,const DAry& src3);

  /**
   * generate random number between 0-1 for every element
   */
  void Random();

  /**
   * set to 1.f if src element < t otherwise set to 0
   * Map(&mask_, [t](float v){return v<=t?1.0f:0.0f;}, mask_);
   */
  void Threshold( const DAry& src, float t);

  int local_shape(int k) const;
  int shape(int k) const;
  const Shape& shape() const;
};

class Shape {
 public:
  const int Size() const;
  /**
   * without the 0-th dimension
   */
  Shape SubShape() const;
};
}  // namespace lapis

#endif  // INCLUDE_DARRAY_DARY_H_
