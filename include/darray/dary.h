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
typdef float* FPtr;
class Shape;
class DAry {
 public:
  DAry():local_size_(0){}
  /**
    * init with the same shape and partition as other, may copy data
    */
  DAry(const DAry& other, bool copy);

  /**
    * create a new dary with data and partition from other dary,
    * but a new shape, this new shape only diff with other dary on
    * the first or last few dimension, total size should the same;
    * like Reshape();
    */
  DAry(const DAry& other, const vector<int>& shape);
  /**
    * set shape and partition as other dary
    */
  void InitLike(const DAry& other);
  /**
    * set shape and partition from proto
    */
  void InitFromProto(const DAryProto& proto);
  void ToProto(DAryProto* proto, bool copyData);

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
    * apply the func for every element in src, put result to dst
    */
  void Map( std::function<float(float)> func, const DAry& src);
  void Map( std::function<float(float, float)> func, const DAry& src1, const DAry& src2);
  void Map( std::function<float(float, float, float)> func, const DAry& src1, const DAry& src2,const DAry& src3);
  /**
    * set to 1.f if src element < t otherwise set to 0
    * Map(&mask_, [t](float v){return v<=t?1.0f:0.0f;}, mask_);
    */
  void Threshold( const DAry& src, float t);


  /**
    * generate random number between 0-1 for every element
    */
  void Random();
  void SampleGaussian(float mean, float std);
  void SampleUniform(float mean, float std);

  void Square( const DAry& src);
  void Copy( const DAry& src);
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
    * sum along dim-th dimension, with range r
    * # of dimensions of dst is that of src-1
    */
  void Sum( const DAry& src, int dim, Range r);

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
    * put max(src,x) into dst
    */
  void Max( const DAry& src, float x);

  /**
    * max element
    */
  float Max();

  void Set(float x);
  /**
    * check whether the element at index is local
    */
  bool isLocal(vector<int> index){
    for (i = 0; i < dim_; i++) {
      if(index[i]>=range_[i].second||index[i]<range_[i].first)
        return false;
    }
    return true;
  }

  /**
    * return the local index range for dim-th dimension
    */
  Range IndexRange(int k) const{
    CHECK_LT(k, dim_];
    return range_[k];
  }

  /**
    * fetch data to local according to index ranges
    * create a new DAry which has the same shape as this dary, but
    * the requested data are local
    */
  DAry FetchData(const vector<Range>& slice) const{
    return DAry(*this, slice);
  }

  /**
    * return the ref for the ary at this index
    * check shape
    */
float& at(int idx0,int idx1, int idx2, int idx3) const {
  return dptr_[locate(idx0,idx1,idx2,idx3)];
}
float& at(int idx0,int idx1, int idx2) const{
  return dptr_[locate(idx0,idx1,idx2)];
}
float& at(int idx0,int idx1) const {
  return dptr_[locate(idx0,idx1)];
}
float& at(int idx0) const {
  return dptr_[locate(idx0)];
}
int locate(int idx0,int idx1, int idx2, int idx3) const {
  CHECK_EQ(dim_,4);
  int pos=((idx0*shape_.s[1]+idx1)*shape_.s[2]+idx2)*shape_.s[3]+idx3;
  CHECK_LT(pos, alloc_size_);
  return pos;
}
int locate(int idx0,int idx1, int idx2) const{
  CHECK_EQ(dim_,3);
  int pos=(idx0*shape_.s[1]+idx1)*shape_.s[2]+idx2;
  CHECK_LT(pos, alloc_size_);
  return pos;
}
float& locate(int idx0,int idx1) const {
  CHECK_EQ(dim_,2);
  int pos=idx0*shape_.s[1]+idx1;
  CHECK_LT(pos, alloc_size_);
  return pos;
}
float& locate(int idx0) const {
  CHECK_EQ(dim_,1);
  CHECK_LT(idx0, alloc_size_);
  return idx0;
}
  /**
    * return the value for the ary at this index
    * check shape
    */
  const float get(int idx0,int idx1, int idx2, int idx3) const {
    return dptr_[locate(idx0,idx1,idx2,idx3)];
  }
  const float get(int idx0,int idx1, int idx2) const{
    return dptr_[locate(idx0,idx1,idx2)];
  }
  const float get(int idx0,int idx1) const{
    return dptr_[locate(idx0,idx1)];
  }
  const float get(int idx0) const{
    return dptr_[locate(idx0)];
  }
  /**
    * allocate memory
    */
  void AllocateMemory();
  void FreeMemory();

  /**
    * set shape if no set before; otherwise check with old shape
    */
  void SetShape(const vector<int>& shape) {
    shape_.Reset(shape);
  }
  void SetShape(const Shape& shape) {
    shape_=shape;
  }
  int shape(int k) const {
    CHECK_LT(k, dim_);
    return shape_.s[k];
  }
  const Shape& shape() const {
    return shape_;
  }
  /**
    * swap dptr
    */
  void SwapDptr(DAry* other) {
    std::swap(dptr_, other->mutable_dptr());
  }
  const FPtr& dptr() const {return dptr_;}
  FPtr& mutable_dptr() const {return dptr_;}
  /**
    * true if memory has allocated or false
    */
  bool allocated() {return alloc_size_>0;}
 protected:
  int dim_;
  Shape shape_;
  vector<Range> range_; // local index range for every dimension
  int alloc_size_; // allocated memory size interms of # of floats
  float* dptr_;
};

class Shape {
 public:
  Shape() {s=nullptr;dim=0;}
  explicit Shape(const Shape& other){
    dim=other.dim;
    s=new int[dim];
    for (i = 0; i < dim; i++) {
      s[i]=other.s[i];
    }
  }
  explicit Shape(const vector<int>& other){
    Reset(other);
  }
  void Reset(const vector<int>& other) {
    if(s!=nullptr)
      delete s;
    dim=other.size();
    s=new int[dim];
    for (i = 0; i < dim; i++) {
      s[i]=other[i];
    }
  }
  const int Size() const;
  /**
   * without the 0-th dimension
   */
  const Shape SubShape() const;
  ~Shape() {delete s;}
 public:
  int* s;
  int dim;
};
}  // namespace lapis

#endif  // INCLUDE_DARRAY_DARY_H_
