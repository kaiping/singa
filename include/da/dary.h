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
class DAry {
 public:
  ~DAry();
  DAry():Ary(), offset_(0), dptr_(nullptr), ga_(nullptr){};//alloc_size_(0),dptr_(nullptr){}
  inline void SetPartition(const Partition& part);
  inline void SetPartition(const short mode, const short dim);
  // alloc local mem; set ga
  void Setup();
  /**
   * init with the same shape and partition as other,
   * if other has no partition, create a local array;
   * alloc memory may copy data
   */
  DAry(const DAry& other, bool copy);
  /**
    * create a new dary with data and partition from other dary,
    * but a new shape, this new shape only diff with other dary on
    * the first or last few dimension, total size should the same;
    * like Reshape();
    */
  DAry Reshape(const DAry& other, const vector<int>& shape) ;
  /**
    * set shape and partition from proto
    */
  void InitFromProto(const DAryProto& proto);
  void ToProto(DAryProto* proto, bool copyData);

  /**
    * set shape and partition as other dary, allocate memory
    */
  void InitLike(const DAry& other);
  /**
    * subdary on the 0-th dim
    */
  DAry operator[](int k) const ;

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
  void SumRow(const DAry& src);
  void SumCol(const DAry& src);

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
    return make_pair<int,int>(part_.lo[k], part_.hi[k]);
  }

  /**
    * fetch data to local according to index ranges
    * create a new DAry which has the same shape as this dary, but
    * the requested data are local
    */
  DAry Fetch(const vector<Range>& slice);


  /**
    * return the ref for the ary at this index
    * check shape
    */
  float* addr(int idx0,int idx1, int idx2, int idx3) const {
    return dptr_+locate(idx0,idx1,idx2,idx3);
  }
  float* addr(int idx0,int idx1, int idx2) const{
    return dptr_+locate(idx0,idx1,idx2);
  }
  float* addr(int idx0,int idx1) const {
    return dptr_+locate(idx0,idx1);
  }
  float* addr(int idx0) const {
    return dptr_+locate(idx0);
  }
  int locate(int idx0,int idx1, int idx2, int idx3) const {
    CHECK_EQ(dim_,4);
    int pos=((idx0*shape_.s[1]+idx1)*shape_.s[2]+idx2)*shape_.s[3]+idx3;
    return pos-ga_==nullptr?0:ga_->offset();
  }
  int locate(int idx0,int idx1, int idx2) const{
    CHECK_EQ(dim_,3);
    int pos=(idx0*shape_.s[1]+idx1)*shape_.s[2]+idx2;
    CHECK(pos> part_.start);
    return pos-ga_==nullptr?0:ga_->offset();
  }
  int locate(int idx0,int idx1) const {
    CHECK_EQ(dim_,2);
    int pos=idx0*shape_.s[1]+idx1;
    CHECK(pos> part_.start);
    return pos-ga_==nullptr?0:ga_->offset();
  }
  int locate(int idx0) const {
    CHECK_EQ(dim_,1);
    return idx0-ga_==nullptr?0:ga_->offset();
  }
  /**
    * return the value for the ary at this index
    * check shape
    */
  float& at(int idx0,int idx1, int idx2, int idx3) const {
    return dptr_[locate(idx0,idx1,idx2,idx3)];
  }
  float& at(int idx0,int idx1, int idx2) const{
    return dptr_[locate(idx0,idx1,idx2)];
  }
  float& at(int idx0,int idx1) const{
    return dptr_[locate(idx0,idx1)];
  }
  float& at(int idx0) const{
    return dptr_[locate(idx0)];
  }
  bool local(){
    return part_.mode==-1;
  }
  bool cached(){
    return dptr_!=nullptr;
  }
  void SetShape(const vector<int>& shape) {
    dim_=shape.size();
    shape_.Reset(shape);
    size_=shape_.Size();
  }
  void SetShape(const Shape& shape) {
    dim_=shape.dim;
    shape_=shape;
    size_=shape.Size();
  }

 protected:
  int offset_;// offset to the base dary
  std::shared_ptr<float> dptr_;
  std::shard_ptr<GAry> ga_;
  std::shard_ptr<DAry> parent_;
  Partition part_;
  Shape shape_;
  static arraymath::ArrayMath& arymath();
};

void DAry::SetPartition(const Partition& part){
  part_=part;
}
void DAry::SetPartition(const vector<Range>& slice){
  part_=slice;
}

}  // namespace lapis

#endif  // INCLUDE_DARRAY_DARY_H_
