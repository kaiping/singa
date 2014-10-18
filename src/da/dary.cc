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
  delete ga;
}

DAry DAry::Setup() {
  CHECK(mode_>=-2);
  if(partition_.mode==-1)
    Cache();
  else
    ga_.Setup(shape_, &part_);
}

DAry DAry::Reshape(const vector<int>& shape) {
  // stride for non continuous
  DAry ret;
  part_.CheckReshape(shape_, shape);
  ret.dptr_=other.dptr_;
  ret.ga_=other.ga_;
  ret.SetShape(shape);
  ret.part_=part_.Repartition(shape);;
  return ret;
}

void DAry::InitFromProto(const DAryProto& proto) {
  vector<int> shape;
  for(auto s: proto.shape())
    shape.push_back(s);
  SetShape(shape);

  part_.mode=proto.mode();
  part_.pdim=proto.pdim();
  for (int i = 0; i < dim_; i++) {
    part_.lo[i]=proto.lo(i);
    part_.hi[i]=proto.hi(i);
  }
  Setup();
}

void DAry::ToProto(DAryProto* proto, bool copyData) {
  for (int i = 0; i < dim_; i++)
    proto->add_shape(shape_.s[i]);
  proto->set_mode(part_.mode);
  proto->set_pdim(part_.pdim);
  for (int i = 0; i < dim_; i++) {
    proto->add_lo(part_.lo[i]);
    proto->add_hi(part_.hi[i]);
  }
}

DAry DAry::operator[](int k) const {
  CHECK(k>=0&&k<shape_.s[0]);
  DAry ret;
  int offset=(k-part_.lo[0])*ret.size_;
  ret.dptr_=dptr_+offset;
  ret.SetShape(shape_.SubShape());
  ret.SetPartition(part_.SubPartition());
  if(ga_!=nullptr)
    ret.ga_=ga_->Sub(offset, ret.size_);
  return ret;
}

DAry::DAry(const DAry& other, bool copy) {
  SetShape(other.shape_);
  Cache();
  range_=other.range_;
  if(copy)
    Copy(other);
}
DAry::DAry(const vector<int>& shape) {
  dptr_=nullptr;
  SetShape(shape);
  Cache();
}
DAry DAry::Fetch(const vector<Range>& slice) {
  if(part_.SameAs(slice)){
      if(!cached())
        Cache();
      ga_->Get(dptr_);
      return *this;
    }
    DAry ret;
    ret.SetShape(shape_);
    ret.SetPartition(slice);
    ret.Cache();
    ret.ga_.Setup(ga_, slice);
    ret.ga_->Get(ret.dptr_);
    return ret;
}

void DAry::InitLike(const DAry& other) {
  SetShape(other.shape_);
  part_=other.part_;
  Cache();
}
DAry DAry::FetchEleOp(const DAry& src) {
  if(!cached())
    Cache();
  return src.Fetch(part_.Slice());
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
  int M=src1.shape(0), N=src2.shape(1), K=src1.shape(1);
  CHECK_EQ(src2.shape(0),K);

  if(src1.glb()&&src2.glb()&&glb()){
    ga_.Dot(src1.ga_, src2.ga_, trans1, trans2);
    return;
  }
  if(!cached())
    Cache();
  if(!src1.cached())
    src1.Fetch({{part_.lo[0], part_.lo[1]}, {0,K}});
  if(!src2.cached())
    src2.Fetch({{0,K},{part_.hi[0], part_.hi[1]}});
  M=part_.lo[1]-part_.lo[0];N=part_.hi[1]-part_.hi[0];
  int lda=trans1==false?K:M;
  int ldb=trans2==false?N:K;
  CBLAS_TRANSPOSE TransA =trans1?CblasTrans:CblasNoTrans;
  CBLAS_TRANSPOSE TransB =trans2?CblasTrans:CblasNoTrans;
  cblas_sgemm(CblasRowMajor,  TransA, TransB, M, N, K,
      1.0f, src1.dptr_, lda, src2.dptr_, ldb, 0.0f, dptr_, N);
  if(!local())
    ga_.Put(dptr_,part_);
}
void DAry::Copy( const DAry& src) {
  CHECK_EQ(size_, src.size_);
  if(src.local()||local()){
    DAry fetched=FetchEleOp(src);
    memcpy(dptr_, fetched.dptr_, alloc_size_*sizeof(float));
    PutIfGlobal();
  }
  else{
    ga_.Copy(src.ga_);
  }
}

void DAry::Mult( const DAry& src1, const DAry& src2) {
  int len=shape_.Size();
  CHECK_EQ(len, src1.shape().Size());
  CHECK_EQ(len, src2.shape().Size());

  if(src1.local()||src2.local()||local()){
    FetchEleOp(src1);
    FetchEleOp(src2);
    arymath().mul(dptr_, src1.dptr_,src2.dptr_, part_.Size());
    PutIfGlobal();
  }
  else{
    ga_.Mult(src1, src2);
  }
}
void DAry::Div( const DAry& src1, const DAry& src2){
  int len=shape_.Size();
  CHECK_EQ(len, src1.shape().Size());
  CHECK_EQ(len, src2.shape().Size());
  if(local()||src1.local()||src2.local()){
  FetchForEleOp(src1);
  FetchForEleOp(src2);
  arymath().div(dptr_, src1.dptr_,src2.dptr_, part_.Size());
  PutIfGlobal();
  }else{
    ga_.Div(src1.ga_, src2.ga_);
  }
}


void DAry::Mult(const DAry& src, const float x) {
  int len=shape_.Size();
  CHECK_EQ(len, src.shape().Size());
  arymath().mul(dptr_, x, src.dptr_, part_.Size());
  PutIfGlobal();
}

void DAry::Div( const DAry& src, const float x) {
  CHECK(x!=0);
  Mult(src, 1.0f/x);
}


/**
  * dst=src1-src2
  */
void DAry::Minus( const DAry& src1, const DAry& src2) {
  int len=shape_.Size();
  CHECK_EQ(len, src1.shape().Size());
  CHECK_EQ(len, src2.shape().Size());
  FetchEleOp(src1);
  FetchEleOp(src2);
  arymath().sub(dptr_, src1.dptr_,src2.dptr_, len);
  PutIfGlobal();
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
  if(local()||src1.local()||src2.local()){
  FetchEleOp(src1);
  FetchEleOp(src2);
  arymath().add(dptr_,src1.dptr_, src2.dptr_, len);
  PutIfGlobal();
  }else{
    ga_.Add(src1.ga_, src2.ga_);
  }
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
  if(local()||src.local()){
  FetchEleOp(src);
  arymath().add(dptr_, x, src.dptr_, len);
  PutIfGlobal();
  }else{
    ga_.Add(src1.ga_,x);
  }
}
void DAry::Add(const DAry& src){
  Add(*this, src);
}

/**
  * generate random number between 0-1 for every element
  */
void DAry::Random() {
  if(!cached())
    Cache();
  arymath().random(dptr_, 0.0f, 1.0f, size_);
  PutIfGlobal();
}
void DAry::SampleGaussian(float mean, float std){
  if(!cached())
    Cache();

  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator(seed);
  std::normal_distribution<double> distribution(mean, std);
  for (int i = 0; i < size_; i++) {
    dptr_[i]=distribution(generator);
  }
  PutIfGlobal();
}
void DAry::SampleUniform(float low, float high) {
  if(!cached())
    Cache();
  arymath().random(dptr_, low, high, size_);
  PutIfGlobal();
}

void DAry::Square(const DAry& src) {
  FetchEleOp(src);
  CHECK_EQ(size_, src.size_);
  arymath().mul(dptr_, src.dptr_, src.dptr_, size_);
  PutIfGlobal();
}

/**
  * dst=src^x
  */
void DAry::Pow( const DAry& src, const float x) {
  CHECK_EQ(size_, src.size_);
  FetchEleOp(src);
  arymath().pow(dptr_, src.dptr_, x, size_);
  PutIfGlobal();
}

/**
  * set to 1.f if src element < t otherwise set to 0
  * Map(&mask_, [t](float v){return v<=t?1.0f:0.0f;}, mask_);
  */
void DAry::Threshold( const DAry& src, float t) {
  FetchEleOp(src);
  Map([t](float v) {return v<=t?1.0f:0.0f;}, src);
  PutIfGlobal();
}
void DAry::Map( std::function<float(float)> func, const DAry& src) {
  CHECK_EQ(size_,src.size_);
  FetchEleOp(src);
  for (int i = 0; i < size_; i++) {
    dptr_[i]=func(src.dptr_[i]);
  }
  PutIfGlobal();
}
void DAry::Map( std::function<float(float, float)> func, const DAry& src1, const DAry& src2) {
  CHECK_EQ(size_, src1.size_);
  FetchEleOp(src1);
  FetchEleOp(src2);
  for (int i = 0; i < size_; i++) {
    dptr_[i]=func(src1.dptr_[i], src2.dptr_[i]);
  }
  PutIfGlobal();
}
void DAry::Map( std::function<float(float, float, float)> func, const DAry& src1, const DAry& src2,const DAry& src3){
  CHECK_EQ(size_, src1.size_);
  FetchEleOp(src1);
  FetchEleOp(src2);
  FetchEleOp(src3);
  for (int i = 0; i < size_; i++) {
    dptr_[i]=func(src1.dptr_[i], src2.dptr_[i], src3.dptr_[i]);
  }
  PutIfGlobal();
}

/**
  * sum along dim-th dimension, with range r
  * # of dimensions of dst is that of src-1
  */
void DAry::Sum( const DAry& src, Range r) {
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
void DAry::SumRow(const DAry& src) {
  CHECK_EQ(dim_,1);
  CHECK_EQ(src.dim_,2);
  CHECK_EQ(size_, src.shape_.s[1]);
  for (int i = 0; i < src.shape_.s[0]; i++) {
    arymath().add(dptr_,src.dptr_+size_*i, dptr_, size_);
  }
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
void DAry::SumCol(const DAry& src) {
  CHECK_EQ(dim_,1);
  CHECK_EQ(src.dim_,2);
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
  if(local())
    return arymath().sum(dptr_, size_);
  else
  return ga_.Sum();
}
/**
  * put max(src,x) into dst
  */
void DAry::Max( const DAry& src, float x) {
  FetchEleOp(src);
  Map([x](float v) {return v<x?x:v;}, src);
  PutIfGlobal();
}

/**
  * max element
  */
float DAry::Max(){
  ga_.Fetch(shape_.ToSlice());
  return arymath().max(dptr_, size_);
}

void DAry::Set(float x){
  ga_.Fill(x, shape_.ToSlice());
  //arymath().fill(dptr_, x, shape_.Size());
}

void DAry::Cache() {
  if(dptr_!=nullptr){
    LOG(WARNING)<<"the dary has been allocated before, size: "<<alloc_size_;
    delete dptr_;
  }
  // it is possible the range is empry on some dimension
  alloc_size_=part_.size();
  dptr_=new float[alloc_size_];
  ga_.Get(dptr_, part_);
}

void DAry::FreeMemory() {
  if(dptr_!=nullptr)
    delete dptr_;
  else
    CHECK(alloc_size_==0)<<"dptr_ is null but alloc_size is "<<alloc_size_;
  alloc_size_=0;
}

DAry::~DAry() {
  FreeMemory();
}
}  // namespace lapis
