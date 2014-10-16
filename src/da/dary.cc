// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-09-10 16:43
#include <cblas.h>
#include <chrono>
#include <random>

#include "darray/arraymath.h"
#include "darray/dary.h"

namespace lapis {
arraymath::ArrayMath& DAry::arymath(){
  static arraymath::ArrayMath am=arraymath::ArrayMath();
  return am;
}
int DAry::allocated_floats_=0;
DAry::DAry(const Shape& shape) {
  dptr_=nullptr;
  alloc_size_=0;
  SetShape(shape);
  AllocateMemory();
}
DAry::DAry(const vector<int>& shape) {
  dptr_=nullptr;
  alloc_size_=0;
  SetShape(shape);
  AllocateMemory();
}

void DAry::InitLike(const DAry& other) {
  SetShape(other.shape_);
  AllocateMemory();
  //range_=other.range_;
}

void DAry::InitFromProto(const DAryProto& proto) {
  vector<int> s;
  for(auto x: proto.shape())
    s.push_back(x);
  SetShape(s);
  //AllocateMemory();
  if(proto.value_size()>0){
    AllocateMemory();
    CHECK_EQ(alloc_size_, proto.value_size());
    for (int i = 0; i < proto.value_size(); i++) {
      dptr_[i]=proto.value(i);
    }
  }
}

void DAry::ToProto(DAryProto* proto, bool copyData) {
  for (int i = 0; i < dim_; i++) {
    proto->add_shape(shape_.s[i]);
    proto->add_range_start(range_[i].first);
    proto->add_range_end(range_[i].second);
  }
}
/**
  * Dot production
  */
void DAry::Dot( const DAry& src1, const DAry& src2, bool trans1, bool trans2){
  CHECK(dptr_!=src1.dptr_);
  CHECK(dptr_!=src2.dptr_);
  CHECK_EQ(src1.dim_,2);
  CHECK_EQ(src2.dim_,2);
  CHECK_EQ(dim_,2);
  int M=trans1==false?src1.shape_.s[0]:src1.shape_.s[1];
  int K=trans2==false?src2.shape_.s[0]:src2.shape_.s[1];
  int N=trans2==false?src2.shape_.s[1]:src2.shape_.s[0];
  CHECK_EQ(shape_.s[0],M);
  CHECK_EQ(shape_.s[1],N);
  CHECK_EQ(K, trans1==false?src1.shape_.s[1]:src1.shape_.s[0]);
  CBLAS_TRANSPOSE TransA =trans1?CblasTrans:CblasNoTrans;
  CBLAS_TRANSPOSE TransB =trans2?CblasTrans:CblasNoTrans;
  cblas_sgemm(CblasRowMajor,  TransA, TransB, M, N, K,
      1.0f, src1.dptr_, src1.shape(1), src2.dptr_, src2.shape(1), 0.0f,
      dptr_, shape_.s[1]);
}
void DAry::Mult( const DAry& src1, const DAry& src2) {
  int len=shape_.Size();
  CHECK_EQ(len, src1.shape().Size());
  CHECK_EQ(len, src2.shape().Size());
  arymath().mul(dptr_, src1.dptr_,src2.dptr_, len);
}
void DAry::Mult( const DAry& src, const float x) {
  int len=shape_.Size();
  CHECK_EQ(len, src.shape().Size());
  arymath().mul(dptr_, x, src.dptr_, len);
}
void DAry::Div( const DAry& src, const float x) {
  CHECK(x!=0);
  Mult(src, 1.0f/x);
}
void DAry::Div( const DAry& src1, const DAry& src2){
  int len=shape_.Size();
  CHECK_EQ(len, src1.shape().Size());
  CHECK_EQ(len, src2.shape().Size());
  arymath().div(dptr_, src1.dptr_,src2.dptr_, len);
}

/**
  * dst=src1-src2
  */
void DAry::Minus( const DAry& src1, const DAry& src2) {
  int len=shape_.Size();
  CHECK_EQ(len, src1.shape().Size());
  CHECK_EQ(len, src2.shape().Size());
  arymath().sub(dptr_, src1.dptr_,src2.dptr_, len);
}

void DAry::Minus( const DAry& src, const float x) {
  Add(src, -x);
}
void DAry::Minus( const DAry& src) {
  Minus(*this, src);
}
/**
  * dst=src1+src2
  */
void DAry::Add( const DAry& src1, const DAry& src2){
  int len=shape_.Size();
  CHECK_EQ(len, src1.shape().Size());
  CHECK_EQ(len, src2.shape().Size());
  arymath().add(dptr_, src1.dptr_,src2.dptr_, len);
}

void DAry::Add(const float x) {
  Add(*this, x);
}
/**
  * dst=src1+x
  */
void DAry::Add( const DAry& src, const float x){
  int len=shape_.Size();
  CHECK_EQ(len, src.shape().Size());
  arymath().add(dptr_, x, src.dptr_, len);
}
void DAry::Add(const DAry& src){
  int len=shape_.Size();
  CHECK_EQ(len, src.shape().Size());
  arymath().add(dptr_, dptr_, src.dptr_, len);
}

/**
  * generate random number between 0-1 for every element
  */
void DAry::Random() {
  arymath().random(dptr_, 0.0f, 1.0f, size_);
}
void DAry::SampleGaussian(float mean, float std){
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator(seed);
  std::normal_distribution<double> distribution(mean, std);
  for (int i = 0; i < size_; i++) {
    dptr_[i]=distribution(generator);
  }
}
void DAry::SampleUniform(float low, float high) {
  arymath().random(dptr_, low, high, size_);
}

void DAry::Square( const DAry& src) {
  CHECK_EQ(size_, src.size_);
  arymath().mul(dptr_, src.dptr_, src.dptr_, size_);
}
void DAry::Copy( const DAry& src) {
  CHECK_EQ(size_, src.size_);
  memcpy(dptr_, src.dptr_, size_*sizeof(float));
}
/**
  * dst=src^x
  */
void DAry::Pow( const DAry& src, const float x) {
  CHECK_EQ(size_, src.size_);
  arymath().pow(dptr_, src.dptr_, x, size_);
}

/**
  * set to 1.f if src element < t otherwise set to 0
  * Map(&mask_, [t](float v){return v<=t?1.0f:0.0f;}, mask_);
  */
void DAry::Threshold( const DAry& src, float t) {
  Map([t](float v) {return v<=t?1.0f:0.0f;}, src);
}
void DAry::Map( std::function<float(float)> func, const DAry& src) {
  CHECK_EQ(size_,src.size_);
  for (int i = 0; i < size_; i++) {
    dptr_[i]=func(src.dptr_[i]);
  }
}
void DAry::Map( std::function<float(float, float)> func, const DAry& src1, const DAry& src2) {
  CHECK_EQ(size_, src1.size_);
  for (int i = 0; i < size_; i++) {
    dptr_[i]=func(src1.dptr_[i], src2.dptr_[i]);
  }
}
void DAry::Map( std::function<float(float, float, float)> func, const DAry& src1, const DAry& src2,const DAry& src3){
  CHECK_EQ(size_, src1.size_);
  for (int i = 0; i < size_; i++) {
    dptr_[i]=func(src1.dptr_[i], src2.dptr_[i], src3.dptr_[i]);
  }
}

/**
  * sum along dim-th dimension, with range r
  * # of dimensions of dst is that of src-1
  */
void DAry::Sum( const DAry& src, int dim, Range r) {
  CHECK_EQ(dim,0);
  CHECK_EQ(size_,src.size_/src.shape_.s[0]);
  Set(0.0f);
  for (int i = r.first; i < r.second; i++) {
    arymath().add(dptr_, dptr_, src.dptr_+i*size_, size_);
  }
}

/**
  * src must be a matrix
  * this is a vector
  */
void DAry::SumRow(const DAry& src, bool reset) {
  CHECK_EQ(dim_,1);
  CHECK_EQ(src.dim_,2);
  CHECK_EQ(size_, src.shape_.s[1]);
  if(reset)
    Set(0.0f);
  for (int i = 0; i < src.shape_.s[0]; i++) {
    arymath().add(dptr_,src.dptr_+size_*i, dptr_, size_);
  }
}

/**
  * Add the src to dst as a vector along dim-th dimension
  * i.e., the dim-th dimension should have the same length as src
  */
void DAry::AddVec(const DAry&src, int dimidx) {
}
void DAry::AddRow( const DAry& src) {
  // only support dim being the last dimension
  CHECK_EQ(dim_, 2);
  CHECK_EQ(src.dim_, 1);
  CHECK_EQ(src.size_, shape_.s[1]);
  for (int i = 0; i < shape_.s[0]; i++) {
    arymath().add(dptr_+src.size_*i, dptr_+src.size_*i, src.dptr_, src.size_);
  }
}

void DAry::AddCol(const DAry& src) {
  CHECK_EQ(src.dim_,1);
  CHECK_EQ(dim_,2);
  CHECK_EQ(src.size_, shape_.s[0]);
  for (int i = 0; i < shape_.s[0]; i++) {
    arymath().add(dptr_+shape_.s[1]*i, src.dptr_[i],dptr_+shape_.s[1]*i, shape_.s[1]);
  }
}
void DAry::SumCol(const DAry& src, bool reset) {
  CHECK_EQ(dim_,1);
  CHECK_EQ(src.dim_,2);
  if(reset)
    Set(0.0f);
  for (int i = 0; i < size_; i++) {
    dptr_[i]+=arymath().sum(src.dptr_+i*src.shape_.s[1], src.shape_.s[1]);
  }
}




/**
  * sum the src except the dim-th dimension
  * e.g., let src be a tensor with shape (2,4,5), then SumExcept(dst, src,1)
  * would results a vector of length 4
  */
void DAry::SumExcept(const DAry& src, int dim){}

/**
  * sum all elements
  */
float DAry::Sum(){
  return arymath().sum(dptr_, size_);
}
/**
  * put max(src,x) into dst
  */
void DAry::Max( const DAry& src, float x) {
  Map([x](float v) {return v<x?x:v;}, src);
}

/**
  * max element
  */
float DAry::Max(){
  return arymath().max(dptr_, size_);
}

void DAry::Set(float x){
  arymath().fill(dptr_, x, shape_.Size());
}

void DAry::AllocateMemory() {
  int size_to_allocate=dim_>0;
  // it is possible the range is empry on some dimension
  CHECK_EQ(dim_, range_.size());
  for (auto& r: range_)
    size_to_allocate*=r.second-r.first>0?r.second-r.first:1;

  if(dptr_!=nullptr){
    if(alloc_size_==size_to_allocate)
      return;
    LOG(WARNING)<<"old size "<<alloc_size_<<" new size "<<size_to_allocate;
    delete dptr_;
  }
  alloc_size_=size_to_allocate;
  dptr_=new float[alloc_size_];
  allocated_floats_+=alloc_size_;
}

void DAry::FreeMemory() {
  if(dptr_!=nullptr)
    delete dptr_;
  else
    CHECK(alloc_size_==0)<<"dptr_ is null but alloc_size is "<<alloc_size_;
  allocated_floats_-=alloc_size_;
  alloc_size_=0;
}

DAry::~DAry() {
  //FreeMemory();
}
}  // namespace lapis
