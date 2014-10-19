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
  delete ga_;
}
void DAry::InitFromProto(const DAryProto& proto) {
  vector<int> shape;
  for(auto s: proto.shape())
    shape.push_back(s);
  shape_.Reset(shape);
  CHECK(proto.offset()==0);
  Setup(proto.mode());
}

void DAry::ToProto(DAryProto* proto, bool copyData) {
  CHECK(offset_==0);
  for (int i = 0; i < dim_; i++)
    proto->add_shape(shape_.s[i]);
  proto->set_mode(part_.mode);
}

DAry DAry::Setup(const int mode) {
  if(mode<0){
    part_.LocalSetup(shape_);
    Allocate();
  }
  else{
    part_.setpDim(mode);
    ga_=std::make_shared(new GAry());
    dptr_=ga_->Setup(shape_, &part_);
  }
}

DAry DAry::Reshape(const vector<int>& shape) {
  // stride for non continuous
  DAry ret;
  ret.offset_=other.offset_;
  ret.dptr_=other.dptr_;
  ret.ga_=other.ga_;
  ret.part_=part_;
  ret.shape_.Reset(shape);
  return ret;
}

const Range IndexRange(int k){
  CHECK(offset==0);
  return ga_->IndexRange(k);
}

DAry DAry::operator[](int k) const {
  CHECK(k>=0&&k<shape_.s[0]);
  DAry ret;
  ret.shape_=shape_.SubShape();
  ret.offset_=offset_+ret.shape_.size*k;
  ret.part_=part_.SubPartition(k*ret.shape_.size, ret.shape_.size);
  ret.ga_=ga_;
  ret.dptr_=part_.GetPtrOffset(k*ret.shape_.size);
  return ret;
}

DAry::DAry(const DAry& other, bool copy) {
  part_=other.part_;
  part_.SetLocal(); //local
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
DAry DAry::Fetch(const vector<Range>& slice) {
  Partition part(slice);
  if(part_==part)
    return *this;
  return Fetch(part);
}
DAry DAry::Fetch(const Partition& part) {
  CHECK(ga_!=nullptr);
  if(part_==part)
    return *this;
  DAry ret;
  ret.part_.setLocal();
  ret.shape_=shape_;
  ret.offset_=offset_;
  ret.ga_=ga_;
  ret.dptr_=ga_->Get(offset_, part);
  return ret;
}

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
  /*
   *
  const DAry lsrc1=src1.Fetch({{part_.lo[0], part_.lo[1]}, {0,K}});
  const DAry lsrc2=src2.Fetch({{0,K},{part_.hi[0], part_.hi[1]}});
  M=part_.lo[1]-part_.lo[0];N=part_.hi[1]-part_.hi[0];
  */
  int lda=trans1==false?K:M;
  int ldb=trans2==false?N:K;
  CBLAS_TRANSPOSE TransA =trans1?CblasTrans:CblasNoTrans;
  CBLAS_TRANSPOSE TransB =trans2?CblasTrans:CblasNoTrans;
  cblas_sgemm(CblasRowMajor,  TransA, TransB, M, N, K,
      1.0f, src1.dptr_, lda, src2.dptr_, ldb, 0.0f, dptr_, N);
  PutIfGlobal();
}

void DAry::Copy( const DAry& src) {
  CHECK_EQ(shape_.size, src.shape_.size);
  auto dptr=src.ga_->Get(part_);
  memcpy(dptr_.get(), dptr.get(), part_.size*sizeof(float));
}

void DAry::Mult( const DAry& src1, const DAry& src2) {
  int len=shape_.Size();
  CHECK_EQ(shape_.size, src1.shape_.size);
  CHECK_EQ(shape_.size, src2.shape_.size);
  auto dptr1=src1.ga_->Get(part_);
  auto dptr2=src2.ga_->Get(part_);
  arymath().mul(dptr_.get(), dptr1.get(),dptr2.get(), part_.size_);
}

void DAry::Div( const DAry& src1, const DAry& src2){
  int len=shape_.Size();
  CHECK_EQ(shape_.size, src1.shape_.size);
  CHECK_EQ(shape_.size, src2.shape_.size);
  auto dptr1=src1.ga_->Get(part_);
  auto dptr2=src2.ga_->Get(part_);
  arymath().div(dptr_.get(), dptr1.get(),dptr2.get(), part_.size_);
}

void DAry::Mult(const DAry& src, const float x) {
  CHECK_EQ(shape_.size, src.shape_.size);
  auto dptr=src.ga_->Get(part_);
  arymath().mul(dptr_.get(), x, dptr.get(), part_.size);
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
  auto dptr1=src1.ga_->Get(part_);
  auto dptr2=src2.ga_->Get(part_);
  arymath().sub(dptr_.get(), dptr1.get(),dptr2.get(), part_.size_);
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
  auto dptr1=src1.ga_->Get(part_);
  auto dptr2=src2.ga_->Get(part_);
  arymath().add(dptr_.get(), dptr1.get(),dptr2.get(), part_.size_);
}

void DAry::Add(const float x) {
  Add(*this, x);
}
/**
  * dst=src1+x
  */
void DAry::Add( const DAry& src, const float x){
  CHECK_EQ(shape_.size, src.shape_.size);
  auto dptr=src.ga_->Get(part_);
  arymath().add(dptr_.get(), x, dptr.get(), part_.size);
}
void DAry::Add(const DAry& src){
  Add(*this, src);
  PutifGlobal();
}

/**
  * generate random number between 0-1 for every element
  */
void DAry::Random() {
  arymath().random(dptr_.get(), 0.0f, 1.0f, part_.size);
}
void DAry::SampleGaussian(float mean, float std){
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator(seed);
  std::normal_distribution<double> distribution(mean, std);
  float *dptr=dptr_.get();
  for (int i = 0; i < part_.size; i++) {
    dptr[i]=distribution(generator);
  }
}
void DAry::SampleUniform(float low, float high) {
  arymath().random(dptr_.get(), low, high, part_.size);
}

void DAry::Square(const DAry& src) {
  CHECK_EQ(shape_.size, src.shape_.size);
  auto dptr=src.ga_->Get(part_);
  arymath().mul(dptr_.get(), dptr.get(), dptr.get(), part_.size);
}

/**
  * dst=src^x
  */
void DAry::Pow( const DAry& src, const float x) {
  CHECK_EQ(shape_.size, src.shape_.size);
  auto dptr=src.ga_->Get(part_);
  arymath().pow(dptr_.get(), dptr.get(), x, part_.size);
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
  float* srcdptr=src.ga_->Get(part_).get();
  float* dptr=dptr_.get();
  for (int i = 0; i < size_; i++) {
    dptr[i]=func(srcdptr[i]);
  }
}
void DAry::Map( std::function<float(float, float)> func, const DAry& src1, const DAry& src2) {
  CHECK_EQ(shape_.size_, src1.shape_.size_);
  CHECK_EQ(shape_.size_, src2.shape_.size_);
  auto dptr1=src1.ga_->Get(part_).get();
  auto dptr2=src2.ga_->Get(part_).get();
  float *dptr=dptr_.get();
  for (int i = 0; i < size_; i++) {
    dptr[i]=func(dptr1[i], dptr2[i]);
  }
}
void DAry::Map( std::function<float(float, float, float)> func, const DAry& src1, const DAry& src2,const DAry& src3){
  CHECK_EQ(shape_.size_, src1.shape_.size_);
  CHECK_EQ(shape_.size_, src2.shape_.size_);
  CHECK_EQ(shape_.size_, src3.shape_.size_);
  auto dptr1=src1.ga_->Get(part_).get();
  auto dptr2=src2.ga_->Get(part_).get();
  auto dptr3=src3.ga_->Get(part_).get();
  float *dptr=dptr_.get();
  for (int i = 0; i < size_; i++) {
    dptr[i]=func(dptr1[i], dptr2[i], dptr3[i]);
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
  float* dptr=dptr_.get();
  float* srcdptr=src.dptr_.get();
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
  float* dptr=dptr_.get();
  float* srcdptr=src.dptr_.get();
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
  float* dptr=dptr_.get(), *srcdptr=src.dptr_.get();
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
  float* dptr=dptr_.get(), *srcdptr=src.dptr_.get();
  for (int i = 0; i < collen; i++) {
    arymath().add(dptr+rowlen*i, srcdptr[i],dptr+rowlen*i, rowlen);
  }
  PutIfGlobal();
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
  return arymath().sum(dptr_.get(), shape_.size);
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
  return arymath().max(dptr_.get(), shape_.size);
}

void DAry::Fill(const float x){
  arymath().fill(dptr_.get(), x, part_.size);
}

void DAry::Allocate() {
  if(dptr_!=nullptr){
    LOG(WARNING)<<"the dary has been allocated before, size: "<<alloc_size_;
    delete dptr_;
  }
  // it is possible the range is empry on some dimension
  dptr_=std::make_shared(new float[part_.size]);
  alloc_size_=part_.size;
}

DAry::~DAry() {
}
}  // namespace lapis
