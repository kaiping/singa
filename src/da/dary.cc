// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-09-10 16:43
#include <cblas.h>
#include <armci.h>

#include <glog/logging.h>
#include <chrono>
#include <random>
#include "da/dary.h"

namespace lapis {
using std::make_pair;
arraymath::ArrayMath& DAry::arymath(){
  static arraymath::ArrayMath am=arraymath::ArrayMath();
  return am;
}
DAry::~DAry(){
  if(alloc_size_>0)
    delete dptr_;
}
void DAry::Allocate() {
  if(dptr_!=nullptr){
    LOG(ERROR)<<"the dary has been allocated before, size: "<<alloc_size_;
    delete dptr_;
  }
  dptr_=new float[part_.size];
  alloc_size_=part_.size;
}

DAry::DAry():offset_(0), alloc_size_(0), dptr_(nullptr), ga_(nullptr){}
/*
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
*/

void DAry::Setup(const int mode) {
  if(mode<0){
    part_.LocalSetup(shape_);
    Allocate();
  }
  else{
    ga_=std::make_shared<GAry>();
    part_.setpDim(mode);
    part_.local=false;
    dptr_=ga_->Setup(shape_, &part_);
    alloc_size_=0;
  }
}

DAry DAry::Reshape(const vector<int>& shape) {
  DAry ret;
  ret.part_=part_;
  ret.offset_=offset_;
  ret.alloc_size_=0;
  ret.ga_=ga_;
  ret.dptr_=dptr_;
  int row=1;
  for(int i=0;i<part_.getpDim();i++)
    row*=shape_.s[i];
  ret.shape_.Reset(shape);
  int nrow=1;
  for(int i=0;i<shape_.dim;i++){
    if(nrow==row){
      ret.part_.setpDim(i);
      break;
    }
    nrow*=shape_.s[i];
  }
  return ret;
}
DAry DAry::operator[](int k) const {
  CHECK(k>=0&&k<shape_.s[0]);
  DAry ret;
  ret.shape_=shape_.SubShape();
  int localoffset=k*ret.shape_.size;
  ret.part_=part_.Intersect(localoffset, ret.shape_.size);
  if(ret.part_.size>0){
    if(localoffset<part_.start){
      ret.dptr_=dptr_;
      CHECK(part_.start-localoffset==ret.part_.start);
    }
    else
      ret.dptr_=dptr_+part_.GetPtrOffset(localoffset);
  }
  ret.offset_=offset_+localoffset;
  if(ret.part_.size==ret.shape_.size){
    ret.part_.LocalSetup(ret.shape_);
    ret.ga_=nullptr;
  }
  else
    ret.ga_=ga_;
  ret.alloc_size_=0;
  return ret;
}

DAry::DAry(DAry&& other) {
  offset_=other.offset_;
  shape_=other.shape_;
  ga_=other.ga_;
  part_=other.part_;
  dptr_=other.dptr_;
  other.dptr_=nullptr;
  other.ga_=nullptr;
  alloc_size_=other.alloc_size_;
}

DAry::DAry(const DAry& other) {
  offset_=other.offset_;
  alloc_size_=0;
  dptr_=other.dptr_;
  ga_=other.ga_;
  part_=other.part_;
  shape_=other.shape_;
}
DAry& DAry::operator=(DAry&& other) {
  offset_=other.offset_;
  shape_=other.shape_;
  ga_=other.ga_;
  part_=other.part_;
  dptr_=other.dptr_;
  alloc_size_=other.alloc_size_;
  other.dptr_=nullptr;
  other.ga_=nullptr;
  other.alloc_size_=0;
 return *this;
}

DAry& DAry::operator=(const DAry& other) {
  offset_=other.offset_;
  alloc_size_=0;
  dptr_=other.dptr_;
  ga_=other.ga_;
  part_=other.part_;
  shape_=other.shape_;
  return *this;
}
DAry::DAry(const DAry& other, bool copy) {
  offset_=other.offset_;
  shape_=other.shape_;
  ga_=other.ga_;
  part_=other.part_;
  Allocate();
}
DAry::DAry(const vector<int>& shape) {
  shape_.Reset(shape);
  part_.LocalSetup(shape_);
  dptr_=nullptr;
  ga_=nullptr;
  alloc_size_=0;
  Allocate();
}

DAry DAry::Fetch(const vector<Range>& slice) const{
  DAry ret;
  Partition tmp(shape_, slice);
  ret.ga_=nullptr;
  ret.dptr_=FetchPtr(tmp);//ga_->Fetch(tmp, offset_);
  ret.shape_.Reset(slice);
  ret.part_.LocalSetup(ret.shape_);
  ret.alloc_size_=ret.dptr_==dptr_?0:ret.part_.size;
  return ret;
}

float* DAry::FetchPtr(const vector<Range>& slice) const{
  Partition part(shape_, slice);
  if(part.size==0)
    return nullptr;
  if(ga_==nullptr||part==part_)
    return dptr_;
  return ga_->Fetch(part, offset_);
}

float* DAry::FetchPtr(const Partition& part) const{
  if(part.size==0)
    return nullptr;
  if(ga_==nullptr||part==part_)
    return dptr_;
  return ga_->Fetch(part, offset_);
}
/**
  * Dot production
  * either all are local or all are global
  */
void DAry::Dot( const DAry& src1, const DAry& src2, bool trans1, bool trans2){
  CHECK(dptr_!=src1.dptr_);
  CHECK(dptr_!=src2.dptr_);
  CHECK_EQ(src1.shape_.dim,2);
  CHECK_EQ(src2.shape_.dim,2);
  CHECK_EQ(shape_.dim,2);
  int M, K, N;
  float  *dptr1=src1.dptr_, *dptr2=src2.dptr_;
  if(ga_!=nullptr){
    //CHECK(src1.ga_!=nullptr&&src2.ga_!=nullptr);
    auto rrng=ga_->IndexRange(0);
    auto crng=ga_->IndexRange(1);
    M=rrng.second-rrng.first;
    N=crng.second-crng.first;
    vector<Range> slice1, slice2;
    if(!trans1){
      slice1=vector<Range>{rrng, make_pair(0, src1.shape_.s[1])};
      K=src1.shape_.s[1];
    }
    else{
      slice1=vector<Range>{make_pair(0,src1.shape_.s[0]),rrng};
      K=src1.shape_.s[0];
    }
    if(!trans2){
      slice2=vector<Range>{make_pair(0, src2.shape_.s[0]), crng};
    }
    else{
      slice2=vector<Range>{crng, make_pair(0, src2.shape_.s[1])};
    }
    dptr1=src1.FetchPtr(slice1);
    dptr2=src2.FetchPtr(slice2);
  }else{
    M=shape_.s[0];
    N=shape_.s[1];
    //M=trans1?src1.shape_.s[1]:src1.shape_.s[0];
    //N=trans2?src2.shape_.s[0]:src2.shape_.s[1];
    K=trans1?src1.shape_.s[0]:src1.shape_.s[1];
  }
  int lda=trans1==false?K:M;
  int ldb=trans2==false?N:K;
  CBLAS_TRANSPOSE TransA =trans1?CblasTrans:CblasNoTrans;
  CBLAS_TRANSPOSE TransB =trans2?CblasTrans:CblasNoTrans;
  cblas_sgemm(CblasRowMajor,  TransA, TransB, M, N, K,
      1.0f, dptr1, lda, dptr2, ldb, 0.0f, dptr_, N);
  src1.Delete(dptr1);
  src2.Delete(dptr2);
}

void DAry::Copy( const DAry& src) {
  CHECK_EQ(shape_.size, src.shape_.size);
  float* dptr=src.FetchPtr(part_);
  memcpy(dptr_, dptr, part_.size*sizeof(float));
  src.Delete(dptr);
}

void DAry::Mult( const DAry& src1, const DAry& src2) {
  CHECK_EQ(shape_.size, src1.shape_.size);
  CHECK_EQ(shape_.size, src2.shape_.size);
  float* dptr1=src1.FetchPtr(part_);
  float* dptr2=src2.FetchPtr(part_);
  arymath().mul(dptr_, dptr1,dptr2, part_.size);
  src1.Delete(dptr1);
  src2.Delete(dptr2);
}

void DAry::Div( const DAry& src1, const DAry& src2){
  CHECK_EQ(shape_.size, src1.shape_.size);
  CHECK_EQ(shape_.size, src2.shape_.size);
  float* dptr1=src1.FetchPtr(part_);
  float* dptr2=src2.FetchPtr(part_);
  arymath().div(dptr_, dptr1,dptr2, part_.size);
  src1.Delete(dptr1);
  src2.Delete(dptr2);
}

/**
  * dst=src1-src2
  */
void DAry::Minus( const DAry& src1, const DAry& src2) {
  CHECK_EQ(shape_.size, src1.shape_.size);
  CHECK_EQ(shape_.size, src2.shape_.size);
  float* dptr1=src1.FetchPtr(part_);
  float* dptr2=src2.FetchPtr(part_);
  arymath().sub(dptr_, dptr1,dptr2, part_.size);
  src1.Delete(dptr1);
  src2.Delete(dptr2);
}
/**
  * dst=src1+src2
  */
void DAry::Add( const DAry& src1, const DAry& src2){
  CHECK_EQ(shape_.size, src1.shape_.size);
  CHECK_EQ(shape_.size, src2.shape_.size);
  float* dptr1=src1.FetchPtr(part_);
  float* dptr2=src2.FetchPtr(part_);;
  arymath().add(dptr_, dptr1, dptr2, part_.size);
  src1.Delete(dptr1);
  src2.Delete(dptr2);
}

void DAry::Mult(const DAry& src, const float x) {
  CHECK_EQ(shape_.size, src.shape_.size);
  float* dptr=src.FetchPtr(part_);
  arymath().mul(dptr_, x, dptr, part_.size);
  src.Delete(dptr);
}

void DAry::Div( const DAry& src, const float x) {
  CHECK(x!=0);
  Mult(src, 1.0f/x);
}
/**
  * dst=src1+x
  */
void DAry::Add( const DAry& src, const float x){
  CHECK_EQ(shape_.size, src.shape_.size);
  float* dptr=src.FetchPtr(part_);
  arymath().add(dptr_, x, dptr, part_.size);
  src.Delete(dptr);
}

void DAry::Minus( const DAry& src, const float x) {
  Add(src, -x);
}
void DAry::Minus( const DAry& src) {
  Minus(*this, src);
}
void DAry::Add(const float x) {
  Add(*this, x);
}
void DAry::Add(const DAry& src){
  Add(*this, src);
}
void DAry::Square(const DAry& src) {
  CHECK_EQ(shape_.size, src.shape_.size);
  float* dptr=src.FetchPtr(part_);
  arymath().mul(dptr_, dptr, dptr,part_.size);
  src.Delete(dptr);
}
/**
  * dst=src^x
  */
void DAry::Pow( const DAry& src, const float x) {
  CHECK_EQ(shape_.size, src.shape_.size);
  float* dptr=src.FetchPtr(part_);
  arymath().pow(dptr_, dptr, x, part_.size);
  src.Delete(dptr);
}
/**
  * generate random number between 0-1 for every element
  */
void DAry::SampleUniform(const float low, const float high) {
  //arymath().random(dptr_, 0.0f, 1.0f, part_.size);
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator(seed);
  std::uniform_real_distribution<double> distribution(low, high);
  for (int i = 0; i < part_.size; i++) {
    dptr_[i]=distribution(generator);
  }
}
void DAry::SampleGaussian(float mean, float std){
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator(seed);
  std::normal_distribution<double> distribution(mean, std);
  for (int i = 0; i < part_.size; i++) {
    dptr_[i]=distribution(generator);
  }
}
void DAry::Random(){
  //arymath().random(dptr_, 0.0f, 1.0f, part_.size);
  SampleUniform(0.0f, 1.0f);
}

void DAry::Fill(const float x){
  arymath().fill(dptr_, x, part_.size);
}

/**
  * set to 1.f if src element < t otherwise set to 0
  * Map(&mask_, [t](float v){return v<=t?1.0f:0.0f;}, mask_);
  */
void DAry::Threshold( const DAry& src, float t) {
  Map([t](float v) {return v<=t?1.0f:0.0f;}, src);
}
/**
  * put max(src,x) into dst
  */
void DAry::Max( const DAry& src, float x) {
  Map([x](float v) {return v<x?x:v;}, src);
}
void DAry::Map( std::function<float(float)> func, const DAry& src) {
  CHECK_EQ(shape_.size,src.shape_.size);
  float* dptr=src.FetchPtr(part_);
  for (int i = 0; i < part_.size; i++) {
    dptr_[i]=func(dptr[i]);
  }
  src.Delete(dptr);
}
void DAry::Map( std::function<float(float, float)> func, const DAry& src1, const DAry& src2) {
  CHECK_EQ(shape_.size, src1.shape_.size);
  CHECK_EQ(shape_.size, src2.shape_.size);
  float* dptr1=src1.FetchPtr(part_);
  float* dptr2=src2.FetchPtr(part_);;
  for (int i = 0; i < part_.size; i++) {
    dptr_[i]=func(src1.dptr_[i], src2.dptr_[i]);
  }
  src1.Delete(dptr1);
  src2.Delete(dptr2);
}
void DAry::Map( std::function<float(float, float, float)> func, const DAry& src1, const DAry& src2,const DAry& src3){
  CHECK_EQ(shape_.size, src1.shape_.size);
  CHECK_EQ(shape_.size, src2.shape_.size);
  CHECK_EQ(shape_.size, src3.shape_.size);
  float* dptr1=src1.FetchPtr(part_);
  float* dptr2=src2.FetchPtr(part_);;
  float* dptr3=src3.FetchPtr(part_);;
  for (int i = 0; i < part_.size; i++) {
    dptr_[i]=func(src1.dptr_[i], src2.dptr_[i], src3.dptr_[i]);
  }
  src1.Delete(dptr1);
  src2.Delete(dptr2);
  src3.Delete(dptr3);
}

/***********************************************
 * Following functions are not safe.
 * e.g., a=b.SumRow(), the b should have all rows, and the column partition
 * should be the same as a.
 * if b is partitioned on 1-th dim, just make column partiton is correct
 * if b is partitioned on 0-th dim, two approaches:
 *    1) fetch all rows, a must only appear on one node
 *    2) do this ops on every node where b is partitioned, then aggregate all a
 **********************************************/
/**
  * sum along dim-th dimension, with range r
  * # of dimensions of dst is that of src-1
  */
void DAry::Sum( const DAry& src, const Range& r) {
  CHECK(ga_==nullptr);
  int rowlen=part_.size;
  CHECK_EQ(rowlen, src.shape_.size/src.shape_.s[0]);
  Fill(0.0f);
  float* dptr=dptr_;
  float* srcdptr=src.dptr_;
  for (int i = r.first; i < r.second; i++) {
    arymath().add(dptr, dptr, srcdptr+i*rowlen, rowlen);
  }
}

/**
  * src must be a matrix
  * this is a vector
  */
void DAry::SumRow(const DAry& src) {
  CHECK(ga_==nullptr);
  CHECK_EQ(shape_.dim,1);
  CHECK_EQ(src.shape_.dim,2);
  Sum(src, make_pair(0, shape_.s[0]));
}
void DAry::SumCol(const DAry& src) {
  CHECK(ga_==nullptr);
  CHECK_EQ(src.shape_.dim,2);
  CHECK_EQ(shape_.dim,1);
  int collen=shape_.size;
  CHECK_EQ(collen, src.shape_.s[0]);
  int rowlen=src.shape_.s[1];
  float* dptr=dptr_, *srcdptr=src.dptr_;
  Fill(0.0f);
  for (int i = 0; i < collen; i++) {
    dptr[i]+=arymath().sum(srcdptr+i*rowlen, rowlen);
  }
}
void DAry::AddRow( const DAry& src) {
  CHECK(ga_==nullptr);
  CHECK_EQ(src.shape_.dim,1);
  CHECK_EQ(shape_.dim,2);
  int rowlen=src.shape_.size;
  CHECK_EQ(rowlen, shape_.s[1]);
  float* dptr=dptr_;
  float* srcdptr=src.dptr_;
  for (int i = 0; i < shape_.s[0]; i++) {
    arymath().add(dptr+rowlen*i, dptr+rowlen*i, srcdptr, rowlen);
  }
}

void DAry::AddCol(const DAry& src) {
  CHECK(ga_==nullptr);
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
  * sum all elements
  */
float DAry::Sum(){
  CHECK(ga_==nullptr);
  return arymath().sum(dptr_, part_.size);
}
/**
  * max element
  */
float DAry::Max(){
  CHECK(ga_==nullptr);
  return arymath().max(dptr_, part_.size);
}

}  // namespace lapis
