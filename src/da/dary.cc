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
DAry::~DAry(){
  if(local_alloc_)
    delete dptr_;
}

void DAry::InitFromProto(const DAryProto& proto) {
  vector<int> shape;
  for(auto s: proto.shape())
    shape.push_back(s);
  shape_.Reset(shape);
  Setup(proto.mode());
}

void DAry::ToProto(DAryProto* proto, bool copyData) {
  CHECK(offset_==0);
  for (int i = 0; i < dim_; i++)
    proto->add_shape(shape_.s[i]);
  proto->set_mode(part_.mode);
}

DAry DAry::Setup(const int mode) {
  if(mode<0)
    Allocate();
  else{
    ga_=std::make_shared(new GAry());
    dptr_=ga_->Setup(shape_, mode);
  }
}

DAry DAry::Reshape(const vector<int>& shape) {
  DAry ret;
  ret.shape_.Reset(shape);
  ret.dptr_=other.dptr_;
  ret.ga_=other.ga_;
  return ret;
}

const Range IndexRange(int k){
  CHECK(ga_!=nullptr);
  return ga_->IndexRange(k);
}

DAry DAry::operator[](int k) const {
  if(ga_!=nullptr){
    k=k-ga_->IndexRange(0).first;
    CHECK(k>=0);
  }
  else
    CHECK(k>=0&&k<shape_.s[0]);
  DAry ret;
  ret.shape_=shape_.SubShape();
  ret.dptr_=dptr_+k*ret.shape_.size;
  return ret;
}

DAry::DAry(const DAry& other, bool copy) {
  shape_=other.shape_;
  ga_=other.ga_;
  Allocate();
}
/*
DAry::DAry(const vector<int>& shape) {
  dptr_=nullptr;
  mode_=1;
  SetShape(shape);
  Allocate();
}
*/
/*
void DAry::InitLike(const DAry& other) {
  mode_=1;
  SetShape(other.shape_);
  Allocate();
}
*/

/**
  * Dot production
  * either all are local or all are global
  */
void DAry::Dot( const DAry& src1, const DAry& src2, bool trans1, bool trans2){
  CHECK(dptr_!=src1.dptr_);
  CHECK(dptr_!=src2.dptr_);
  CHECK_EQ(src1.dim_,2);
  CHECK_EQ(src2.dim_,2);
  CHECK_EQ(dim_,2);
  int M=src1.shape(0), N=src2.shape(1), K=src1.shape(1);
  CHECK_EQ(src2.shape(0),K);
  float  *dptr1=src1.dptr_, *dptr2=src2.dptr_;
  if(ga_!=nullptr){
    CHECK(src1.ga_!=nullptr&&src2.ga_!=nullptr);
    auto rrng=ga_->IndexRange(0);
    auto crng=ga_->IndexRange(1);
    vector<Range> slice1, slice2;
    if(!trans1)
      slice1=vector<Range>{rrng,  {0, src1.shape_.s[1]}};
    else
      slice1=vector<Range>{{0, src1.shape_.s[1]},rrng};
    if(!trans2)
      slice2=vector<Range>{{0, src2.shape_.s[0]}, crng};
    else
      slice2=vector<Range>{crng, {0, src2.shape_.s[0]}};
    dptr1=src1.Get(slice1);
    dptr2=src2.Get(slice2);
  }
  int lda=trans1==false?K:M;
  int ldb=trans2==false?N:K;
  CBLAS_TRANSPOSE TransA =trans1?CblasTrans:CblasNoTrans;
  CBLAS_TRANSPOSE TransB =trans2?CblasTrans:CblasNoTrans;
  cblas_sgemm(CblasRowMajor,  TransA, TransB, M, N, K,
      1.0f, dptr1, lda, dptr2, ldb, 0.0f, dptr_, N);
  if(dptr1!=src1.dptr_)
    delete dptr1;
  if(dptr2!=src2.dptr_)
    delete dptr2;
}

void DAry::Copy( const DAry& src) {
  CHECK_EQ(shape_.size, src.shape_.size);
  //auto dptr=src.ga_->Get(part_);
  int size=ga_==nullptr?shape_.size:ga_->local_size();
  memcpy(dptr_, src.dptr_, size*sizeof(float));
}

void DAry::Mult( const DAry& src1, const DAry& src2) {
  int len=shape_.Size();
  CHECK_EQ(shape_.size, src1.shape_.size);
  CHECK_EQ(shape_.size, src2.shape_.size);
  int size=ga_==nullptr?shape_.size:ga_->local_size();
  arymath().mul(dptr_, src1.dptr_,src2.dptr_, size);
}

void DAry::Div( const DAry& src1, const DAry& src2){
  int len=shape_.Size();
  CHECK_EQ(shape_.size, src1.shape_.size);
  CHECK_EQ(shape_.size, src2.shape_.size);
  int size=ga_==nullptr?shape_.size:ga_->local_size();
  arymath().div(dptr_, src1.dptr_,src2.dptr_, size);
}

void DAry::Mult(const DAry& src, const float x) {
  CHECK_EQ(shape_.size, src.shape_.size);
  int size=ga_==nullptr?shape_.size:ga_->local_size();
  arymath().mul(dptr_, x, src.dptr_, size);
}

void DAry::Div( const DAry& src, const float x) {
  CHECK(x!=0);
  Mult(src, 1.0f/x);
}

/**
  * dst=src1-src2
  */
void DAry::Minus( const DAry& src1, const DAry& src2) {
  CHECK_EQ(shape_.size, src1.shape_.size);
  CHECK_EQ(shape_.size, src2.shape_.size);
  int size=ga_==nullptr?shape_.size:ga_->local_size();
  arymath().sub(dptr_,src1.dptr_,src2.dptr_, size);
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
  CHECK_EQ(shape_.size, src1.shape_.size);
  CHECK_EQ(shape_.size, src2.shape_.size);
  int size=ga_==nullptr?shape_.size:ga_->local_size();
  arymath().add(dptr_, src1.dptr_,src2.dptr_, size);
}

void DAry::Add(const float x) {
  Add(*this, x);
}
/**
  * dst=src1+x
  */
void DAry::Add( const DAry& src, const float x){
  CHECK_EQ(shape_.size, src.shape_.size);
  int size=ga_==nullptr?shape_.size:ga_->local_size();
  arymath().add(dptr_, x, src.dptr_, size);
}
void DAry::Add(const DAry& src){
  Add(*this, src);
  PutifGlobal();
}

/**
  * generate random number between 0-1 for every element
  */
void DAry::Random() {
  int size=ga_==nullptr?shape_.size:ga_->local_size();
  arymath().random(dptr_, 0.0f, 1.0f, size);
}
void DAry::SampleGaussian(float mean, float std){
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator(seed);
  std::normal_distribution<double> distribution(mean, std);
  int size=ga_==nullptr?shape_.size:ga_->local_size();
  for (int i = 0; i < size; i++) {
    dptr_[i]=distribution(generator);
  }
}
void DAry::SampleUniform(float low, float high) {
  int size=ga_==nullptr?shape_.size:ga_->local_size();
  arymath().random(dptr_, low, high, size);
}

void DAry::Square(const DAry& src) {
  CHECK_EQ(shape_.size, src.shape_.size);
  int size=ga_==nullptr?shape_.size:ga_->local_size();
  arymath().mul(dptr_, src.dptr_, src.dptr_, size);
}

/**
  * dst=src^x
  */
void DAry::Pow( const DAry& src, const float x) {
  CHECK_EQ(shape_.size, src.shape_.size);
  int size=ga_==nullptr?shape_.size:ga_->local_size();
  arymath().pow(dptr_, src.dptr_, x, size);
}

/**
  * set to 1.f if src element < t otherwise set to 0
  * Map(&mask_, [t](float v){return v<=t?1.0f:0.0f;}, mask_);
  */
void DAry::Threshold( const DAry& src, float t) {
  Map([t](float v) {return v<=t?1.0f:0.0f;}, src);
}
void DAry::Map( std::function<float(float)> func, const DAry& src) {
  CHECK_EQ(shape_.size,src.shape_.size);
  for (int i = 0; i < size_; i++) {
    dptr_[i]=func(src.dptr_[i]);
  }
}
void DAry::Map( std::function<float(float, float)> func, const DAry& src1, const DAry& src2) {
  CHECK_EQ(shape_.size_, src1.shape_.size_);
  CHECK_EQ(shape_.size_, src2.shape_.size_);
  int size=ga_==nullptr?shape_.size:ga_->local_size();
  for (int i = 0; i < size; i++) {
    dptr_[i]=func(src1.dptr_[i], src2.dptr_[i]);
  }
}
void DAry::Map( std::function<float(float, float, float)> func, const DAry& src1, const DAry& src2,const DAry& src3){
  CHECK_EQ(shape_.size_, src1.shape_.size_);
  CHECK_EQ(shape_.size_, src2.shape_.size_);
  CHECK_EQ(shape_.size_, src3.shape_.size_);
  int size=ga_==nullptr?shape_.size:ga_->local_size();
  for (int i = 0; i < size; i++) {
    dptr_[i]=func(src1.dptr_[i], src2.dptr_[i], src3.dptr_[i]);
  }
}

/**
  * sum along dim-th dimension, with range r
  * # of dimensions of dst is that of src-1
  */
void DAry::Sum( const DAry& src, Range r) {
  int rowsize=shape_.size;
  CHECK_EQ(rowsize, src.shape_.size/src.shape_.s[0]);
  Fill(0.0f);
  float* dptr=dptr_;
  float* srcdptr=src.dptr_;
  for (int i = 0; i < r.second-r.first; i++) {
    arymath().add(dptr, dptr, srcdptr+i*rowsize, rowsize);
  }
}

/**
  * src must be a matrix
  * this is a vector
  */
void DAry::SumRow(const DAry& src) {
  CHECK_EQ(shape_.dim,1);
  CHECK_EQ(src.shape.dim,2);
  Sum(src, {0, shape_.s[0]});
}
void DAry::AddRow( const DAry& src) {
  // only support dim being the last dimension
  CHECK_EQ(src.shape.dim,1);
  CHECK(src.part_.isLocal());
  CHECK_EQ(shape_.dim,2);
  int rowlen=src.shape_.size;
  CHECK_EQ(rowlen, shape_.s[1]);
  float* dptr=dptr_;
  float* srcdptr=src.dptr_;
  for (int i = 0; i < shape_.s[0]; i++) {
    arymath().add(dptr+rowlen*i, dptr+rowlen*i, srcdptr, rowlen);
  }
}
void DAry::SumCol(const DAry& src) {
  CHECK_EQ(src.shape_.dim,2);
  CHECK(src.part_.isLocal());
  CHECK_EQ(shape_.dim,1);
  int collen=shape_.size;
  CHECK_EQ(collen, src.shape_.s[0]);
  int rowlen=src.shape_.s[1];
  float* dptr=dptr_, *srcdptr=src.dptr_;
  Fill(0.0f);
  for (int i = 0; i < collen_; i++) {
    dptr[i]+=arymath().sum(srcdptr+i*rowlen, rowlen);
  }
}
void DAry::AddCol(const DAry& src) {
  CHECK_EQ(src.shape_.dim,1);
  CHECK_EQ(shape_.dim,2);
  int collen=shape_.s[0];
  int rowlen=shape_.s[1];
  CHECK_EQ(src.shape_.size, collen);
  float* dptr=dptr_, *srcdptr=src.dptr_;
  for (int i = 0; i < collen; i++) {
    arymath().add(dptr+rowlen*i, srcdptr[i],dptr+rowlen*i, rowlen);
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
  int size=ga_==nullptr?shape_.size:ga_->local_size();
  return arymath().sum(dptr_, shape_.size);
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
  int size=ga_==nullptr?shape_.size:ga_->local_size();
  return arymath().max(dptr_, shape_.size);
}

void DAry::Fill(const float x){
  int size=ga_==nullptr?shape_.size:ga_->local_size();
  arymath().fill(dptr_, x, shape_.size);
}

void DAry::Allocate() {
  if(dptr_!=nullptr){
    LOG(WARNING)<<"the dary has been allocated before, size: "<<alloc_size_;
    delete dptr_;
  }
  // it is possible the range is empry on some dimension
  dptr_=new float[shape_.size];
  local_alloc_=true;
}
}  // namespace lapis
