// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-09-10 16:34

#ifndef INCLUDE_DARRAY_DARY_H_
#define INCLUDE_DARRAY_DARY_H_

#include <glog/logging.h>

#include <vector>
#include <utility>
#include <functional>
#include <sstream>
#include <string>
#include "darray/arraymath.h"


#include "proto/model.pb.h"

using std::string;
using std::vector;

namespace lapis {
using Range=std::pair<int, int>;

class Shape {
 public:
  Shape():s(), dim(0){}
  Shape(const Shape& other){
    dim=other.dim;
    s[0]=other.s[0]; s[1]=other.s[1]; s[2]=other.s[2]; s[3]=other.s[3];
  }
  Shape(const Shape&& other){
    dim=other.dim;
    s[0]=other.s[0]; s[1]=other.s[1]; s[2]=other.s[2]; s[3]=other.s[3];
  }
  Shape& operator=(const Shape&& other) {
    dim=other.dim;
    s[0]=other.s[0]; s[1]=other.s[1]; s[2]=other.s[2]; s[3]=other.s[3];
    return *this;
  }
  Shape& operator=(const Shape& other) {
    dim=other.dim;
    s[0]=other.s[0]; s[1]=other.s[1]; s[2]=other.s[2]; s[3]=other.s[3];
    return *this;
  }
  Shape(const vector<int>& other){
    Reset(other);
  }
  void Reset(const vector<int>& other) {
    dim=other.size();
    for (unsigned int i = 0; i < other.size(); i++) {
      s[i]=other[i];
    }
  }
  const int Size() const{
    int count=dim>0;
    for (int i = 0; i < dim; i++) {
      count *=s[i];
    }
    return count;
  }
  /**
    * without the 0-th dimension
    */

  const Shape SubShape() const {
    CHECK(dim>1);
    Shape ret;
    ret.dim=dim-1;
    for (int i = 0; i < dim-1; i++) {
      ret.s[i]=s[i+1];
    }
    return ret;
  }

 public:
  int s[4];
  int dim;
};
class DAry {
 public:
  ~DAry();
  DAry():dim_(0), size_(0),alloc_size_(0),dptr_(nullptr){}
  /**
   * init with the same shape and partition as other, alloc memory
   * may copy data
    DAry(const DAry& other);
   */
  /**
   * init based on the shape, alloc memory
   */
  DAry(const vector<int>& shape);
  DAry(const Shape& shape);

  /**
    * create a new dary with data and partition from other dary,
    * but a new shape, this new shape only diff with other dary on
    * the first or last few dimension, total size should the same;
    * like Reshape();
    */
  DAry(const DAry& other, const vector<int>& shape) {
    dptr_=other.dptr_;
    SetShape(shape);
    CHECK_EQ(other.size_, size_);
    alloc_size_=other.alloc_size_;
  }

  // TODO fetch remote data alloc memory
  DAry(const DAry& other, const vector<Range>& slice) {
    dptr_=other.dptr_;
    SetShape(other.shape_);
    range_=slice;
    alloc_size_=size_;
  }

  /**
    * set shape and partition as other dary, allocate memory
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
  DAry operator[](int k) const {
    DAry ret;
    ret.SetShape(shape_.SubShape());
    ret.dptr_=dptr_+k*ret.size_;
    ret.alloc_size_=ret.size_;
    return ret;
  }

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
  void Minus( const DAry& src, const float x) ;
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
  void Add(const float x);
  /**
    * dst=src1+x
    */
  void Add( const DAry& src1, const float x);
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
  void AddVec(const DAry& src, int dimidx);
  void AddRow(const DAry& src);
  void AddCol(const DAry& src);

  /**
    * sum along dim-th dimension, with range r
    * # of dimensions of dst is that of src-1
    */
  void Sum( const DAry& src, int dim, Range r);

  /**
    * src must be a matrix
    * this is a vector
    */
  void SumRow(const DAry& src, bool reset=true);
  void SumCol(const DAry& src, bool reset=true);

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
    * apply the func for every element in src, put result to dst
    */
  void Map(std::function<float(float)> func, const DAry& src);
  void Map(std::function<float(float, float)> func, const DAry& src1, const DAry& src2);
  void Map(std::function<float(float, float, float)> func, const DAry& src1, const DAry& src2,const DAry& src3);

  /**
    * check whether the element at index is local
    */
  bool isLocal(vector<int> index){
    for (int i = 0; i < dim_; i++) {
      if(index[i]>=range_[i].second||index[i]<range_[i].first)
        return false;
    }
    return true;
  }

  /**
    * return the local index range for dim-th dimension
    */
  Range IndexRange(int k) const{
    CHECK(k< dim_);
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
    //CHECK_EQ(dim_,4);
    int pos=((idx0*shape_.s[1]+idx1)*shape_.s[2]+idx2)*shape_.s[3]+idx3;
    //CHECK(pos< alloc_size_);
    return pos;
  }
  int locate(int idx0,int idx1, int idx2) const{
    CHECK_EQ(dim_,3);
    int pos=(idx0*shape_.s[1]+idx1)*shape_.s[2]+idx2;
    CHECK(pos< alloc_size_);
    return pos;
  }
  int locate(int idx0,int idx1) const {
    CHECK_EQ(dim_,2);
    int pos=idx0*shape_.s[1]+idx1;
    CHECK(pos< alloc_size_);
    return pos;
  }
  int locate(int idx0) const {
    CHECK_EQ(dim_,1);
    CHECK(idx0< alloc_size_);
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
    dim_=shape.size();
    shape_.Reset(shape);
    size_=shape_.Size();
    // range is the whole dimension for no partition
    range_.clear();
    for (int i = 0; i < dim_; i++) {
      range_.push_back(std::make_pair(0, shape_.s[i]));
    }
  }
  void SetShape(const Shape& shape) {
    dim_=shape.dim;
    shape_=shape;
    size_=shape.Size();
    range_.clear();
    for (int i = 0; i < dim_; i++) {
      range_.push_back(std::make_pair(0, shape_.s[i]));
    }
  }
  int shape(int k) const {
    CHECK(k< dim_);
    return shape_.s[k];
  }
  const Shape& shape() const {
    return shape_;
  }
  /**
    * swap dptr
    */
  void SwapDptr(DAry* other) {
    std::swap(dptr_, other->dptr_);
  }
  float* dptr() const {return dptr_;}
  /**
    * true if memory has allocated or false
    */
  bool allocated() {return alloc_size_>0;}
  string ToString() {
    std::stringstream ss;
    for (int i = 0; i < size_; i++) {
      ss<<dptr_[i]<<" ";
    }
    return ss.str();
  }
  int size() {return size_;}
  static int allocated_floats() {return allocated_floats_;}
  static arraymath::ArrayMath& arymath();
 protected:
  int dim_;
  int size_;
  int alloc_size_; // allocated memory size interms of # of floats
  float* dptr_;
  Shape shape_;
  vector<Range> range_; // local index range for every dimension

  static int allocated_floats_;
};

}  // namespace lapis

#endif  // INCLUDE_DARRAY_DARY_H_
