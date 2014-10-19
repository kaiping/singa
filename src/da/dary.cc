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
  /*
  if(part_.isLocal()&&!part_.isCache())
    delete dptr_;
    */
}

DAry DAry::Setup(const int mode) {
  if(mode<0)
    part_.setLocal();
  else
    part_.setpDim(mode);
  part_.setDim(shape_.dim);
  if(part.isLocal()){
    part_.lo[0]=part_.lo[1]=part_.lo[2]=part_.lo[3]=0;
    part_.hi[0]=shape_.s[0];part_.hi[1]=shape_.s[1];
    part_.hi[2]=shape_.s[2];part_.hi[3]=shape_.s[3];
    part_.ld[0]=part_.hi[1];part_.ld[1]=part_.hi[2];ld[2]=part_.hi[3];
    Allocate();
  }
  else{
    ga_=std::make_shared(new GAry());
    part_=ga_->Setup(shape_, &part_);
    Allocate();
    ga_->Get(dptr_.get(), part_.lo, part_.hi, part_.ld);
  }
}
DAry DAry::Setup(const Partition& part) {
  part_=part;
  if(part.isLocal()){
    part_.lo[0]=part_.lo[1]=part_.lo[2]=part_.lo[3]=0;
    part_.hi[0]=shape_.s[0];part_.hi[1]=shape_.s[1];
    part_.hi[2]=shape_.s[2];part_.hi[3]=shape_.s[3];
    part_.ld[0]=part_.hi[1];part_.ld[1]=part_.hi[2];ld[2]=part_.hi[3];
    Allocate();
  }
  else{
    ga_=std::make_shared(new GAry());
    part_=ga_->Setup(shape_, part_);
    Allocate();
    ga_->Get(dptr_.get(), part_.lo, part_.hi, part_.ld);
  }
}

DAry DAry::Reshape(const vector<int>& shape) {
  // stride for non continuous
  DAry ret;
  ret.offset_=other.offset_;
  ret.dptr_=other.dptr_;
  ret.ga_=other.ga_;
  ret.part_=part_;
  ret.SetShape(shape);
  return ret;
}

void DAry::InitFromProto(const DAryProto& proto) {
  vector<int> shape;
  for(auto s: proto.shape())
    shape.push_back(s);
  SetShape(shape);
  CHECK(proto.offset()==0);
  part_.mode=proto.mode();
  /*
  for (int i = 0; i < dim_; i++) {
    part_.lo[i]=proto.lo(i);
    part_.hi[i]=proto.hi(i);
  }
  */
  Setup(part);
}

const Range IndexRange(int d){
  CHECK(offset_==0);
  return make_pair<int,int> (part_.lo[d], part_.hi[d]);
}

void DAry::ToProto(DAryProto* proto, bool copyData) {
  CHECK(offset_==0);
  for (int i = 0; i < dim_; i++)
    proto->add_shape(shape_.s[i]);
  proto->set_mode(part_.mode);
  /*
  for (int i = 0; i < dim_; i++) {
    proto->add_lo(part_.lo[i]);
    proto->add_hi(part_.hi[i]);
  }
  */
}

DAry DAry::operator[](int k) const {
  CHECK(k>=0&&k<shape_.s[0]);
  DAry ret;
  ret.SetShape(shape_.SubShape());
  ret.offset_=offset_+k*ret.size_;
  ret.ga_=ga_;
  ret.part_=part_;
  /*
  if(ga_==nullptr){
    ret.dptr_=dptr_+k*ret.size_;
  }
  else{
  */
  ga_->UpdatePartition(&ret.part_, offset_, ret.size);
  //ret.dptr_=ga_->GetPartitionPtr(part_);
  Allocate();
  ga_->Get(dptr_.get(), part_.lo, part_.hi, part_.ld);
  return ret;
}

DAry::DAry(const DAry& other, bool copy) {
  part_=other.part_;
  part_.SetLocal(); //local
  SetShape(other.shape_);
  ga_=other.ga_;
  Allocate();
  ga_->Get(dptr_.get(), part_.lo, part_.hi, part_.ld);
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
  Partition part;
  part.Setup(slice);
  return Fetch(part);
}
DAry DAry::Fetch(const Partition& part) {
  CHECK(ga_!=nullptr);
  if(part_==part)
    return *this;
  DAry ret;
  ret.part_.setLocal();
  ret.SetShape(shape_);
  ret.offset_=offset_+shape_.Offset(ret.part_);
  ret.Allocate();
  ret.ga_=ga_;
  ga_->UpdatePartition(&ret.part_, ret.offset_, ret.shape_.size);
  ga_->Get(ret.dptr_.get(), ret.part_.lo, ret.part_.hi, ret.part_.ld);
  /*
   * conver slice to partition, offset=offset_+slice.offset
    ga_->Fetch(ret.dptr_, lo, hi, ld);
   */
  return ret;
}

/*
void DAry::InitLike(const DAry& other) {
  mode_=1;
  SetShape(other.shape_);
  Allocate();
}
*/

DAry DAry::FetchEleOp(const DAry& src) {
  ShapeCheck(shape_, src.shape_);
  Partition p=ga_->LocateParition();
  return src.Fetch(p);
}

void PutIfGlobal(){
  if(!part_.isLocal())
    ga_->Put(dptr_.get(), part_.lo, part_.hi, part.ld);
}

/**
  * Dot production
  * either all are cached or all are global
  */
void DAry::Dot( const DAry& src1, const DAry& src2, bool trans1, bool trans2){
  CHECK(dptr_!=src1.dptr_);
  CHECK(dptr_!=src2.dptr_);
  CHECK_EQ(src1.dim_,2);
  CHECK_EQ(src2.dim_,2);
  CHECK_EQ(dim_,2);
  int M=src1.shape(0), N=src2.shape(1), K=src1.shape(1);
  CHECK_EQ(src2.shape(0),K);
  const float* dptr1=nullptr, *dptr2=nullptr;
  if(!src1.Cached()&&!src2.Cached()){
    const DAry lsrc1=src1.Fetch({{part_.lo[0], part_.lo[1]}, {0,K}});
    const DAry lsrc2=src2.Fetch({{0,K},{part_.hi[0], part_.hi[1]}});
    M=part_.lo[1]-part_.lo[0];N=part_.hi[1]-part_.hi[0];
    dptr1=lsrc1.dptr_; dptr2=lsrc2.dptr_;
  }else{
    dptr1=src1.dptr_; dptr2=src2.dptr_;
  }
  int lda=trans1==false?K:M;
  int ldb=trans2==false?N:K;
  CBLAS_TRANSPOSE TransA =trans1?CblasTrans:CblasNoTrans;
  CBLAS_TRANSPOSE TransB =trans2?CblasTrans:CblasNoTrans;
  cblas_sgemm(CblasRowMajor,  TransA, TransB, M, N, K,
      1.0f, lsrc1.dptr_, lda, lsrc2.dptr_, ldb, 0.0f, dptr_, N);
  PutIfGlobal();
}

void DAry::Copy( const DAry& src) {
  CHECK_EQ(shape_.size, src.shape_.size);

  const DAry lsrc=src.Fetch(part_);
  memcpy(dptr_.get(), lsrc.dptr_.get(), part_.size*sizeof(float));
  PutifGlobal();
}

void DAry::Mult( const DAry& src1, const DAry& src2) {
  int len=shape_.Size();
  CHECK_EQ(shape_.size, src1.shape_.size);
  CHECK_EQ(shape_.size, src2.shape_.size);

  const DAry lsrc1=src1.Fetch(part_);
  const DAry lsrc2=src2.Fetch(part_);
  arymath().mul(dptr_.get(), lsrc1.dptr_.get(),lsrc2.dptr_.get(), part_.size_);
  PutifGlobal();
}
void DAry::Div( const DAry& src1, const DAry& src2){
  int len=shape_.Size();
  CHECK_EQ(shape_.size, src1.shape_.size);
  CHECK_EQ(shape_.size, src2.shape_.size);

  const DAry lsrc1=src1.Fetch(part_);
  const DAry lsrc2=src2.Fetch(part_);
  arymath().div(dptr_.get(), lsrc1.dptr_.get(),lsrc2.dptr_.get(), part_.size_);
  PutifGlobal();
}


void DAry::Mult(const DAry& src, const float x) {
  CHECK_EQ(shape_.size, src.shape_.size);
  const DAry lsrc=src.Fetch(part_);
  arymath().mul(dptr_.get(), x, lsrc.dptr_.get(), part_.size);
  PutifGlobal();
}

void DAry::Div( const DAry& src, const float x) {
  CHECK(x!=0);
  Mult(src, 1.0f/x);
  PutifGlobal();
}

/**
  * dst=src1-src2
  */
void DAry::Minus( const DAry& src1, const DAry& src2) {
  CHECK_EQ(shape_.size, src1.shape_.size);
  CHECK_EQ(shape_.size, src2.shape_.size);

  const DAry lsrc1=src1.Fetch(part_);
  const DAry lsrc2=src2.Fetch(part_);
  arymath().sub(dptr_.get(), lsrc1.dptr_.get(),lsrc2.dptr_.get(), part_.size_);
  PutifGlobal();
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

  const DAry lsrc1=src1.Fetch(part_);
  const DAry lsrc2=src2.Fetch(part_);
  arymath().add(dptr_.get(), lsrc1.dptr_.get(),lsrc2.dptr_.get(), part_.size_);
  PutifGlobal();
}

void DAry::Add(const float x) {
  Add(*this, x);
}
/**
  * dst=src1+x
  */
void DAry::Add( const DAry& src, const float x){
  CHECK_EQ(shape_.size, src.shape_.size);
  const DAry lsrc=src.Fetch(part_);
  arymath().add(dptr_.get(), x, lsrc.dptr_.get(), part_.size);
  PutifGlobal();
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
  PutifGlobal();
}
void DAry::SampleGaussian(float mean, float std){
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator(seed);
  std::normal_distribution<double> distribution(mean, std);
  float *dptr=dptr.get();
  for (int i = 0; i < part_.size; i++) {
    dptr[i]=distribution(generator);
  }
  PutifGlobal();
}
void DAry::SampleUniform(float low, float high) {
  arymath().random(dptr_.get(), low, high, part_.size);
  PutifGlobal();
}

void DAry::Square(const DAry& src) {
  CHECK_EQ(shape_.size, src.shape_.size);
  const DAry lsrc=src.Fetch(part_);
  arymath().mul(dptr_.get(), lsrc.dptr_.get(), lsrc.dptr_.get(), part_.size);
  PutifGlobal();
}

/**
  * dst=src^x
  */
void DAry::Pow( const DAry& src, const float x) {
  CHECK_EQ(shape_.size, src.shape_.size);
  const DAry lsrc=src.Fetch(part_);
  arymath().pow(dptr_.get(), src.dptr_.get(), x, part_.size);
  PutifGlobal();
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
  DAry lsrc=src.Fetch(part_);
  float *dptr=dptr_.get(), srcdptr=src.dptr_.get();
  for (int i = 0; i < size_; i++) {
    dptr[i]=func(src.dptr[i]);
  }
  PutifGlobal();
}
void DAry::Map( std::function<float(float, float)> func, const DAry& src1, const DAry& src2) {
  CHECK_EQ(shape_.size_, src1.shape_.size_);
  CHECK_EQ(shape_.size_, src2.shape_.size_);
  DAry lsrc1=src1.Fetch(part_);
  DAry lsrc2=src2.Fetch(part_);
  float *dptr=dptr_.get(), dptr1=src1.dptr_.get(), dptr2=src2.dptr_.get();
  for (int i = 0; i < size_; i++) {
    dptr[i]=func(dptr1[i], dptr2[i]);
  }
  PutIfGlobal();
}
void DAry::Map( std::function<float(float, float, float)> func, const DAry& src1, const DAry& src2,const DAry& src3){
  CHECK_EQ(shape_.size_, src1.shape_.size_);
  CHECK_EQ(shape_.size_, src2.shape_.size_);
  CHECK_EQ(shape_.size_, src3.shape_.size_);
  DAry lsrc1=src1.Fetch(part_);
  DAry lsrc2=src2.Fetch(part_);
  DAry lsrc3=src3.Fetch(part_);
  float *dptr=dptr_.get(), dptr1=src1.dptr_.get(), dptr2=src2.dptr_.get(), dptr3=src3.dptr_.get();
  for (int i = 0; i < size_; i++) {
    dptr[i]=func(dptr1[i], dptr2[i], dptr3[i]);
  }
  PutIfGlobal();
}

/**
  * sum along dim-th dimension, with range r
  * # of dimensions of dst is that of src-1
  */
void DAry::Sum( const DAry& src, Range r) {
  int rowsize=shape_.size;
  CHECK_EQ(rowsize, src.shape_.size/src.shape_.s[0]);
  SetLocal(0.0f);

  vector<Range> slice;
  slice.push_back(r);
  for (int i = 1; i < src.shape_.dim; i++)
    slice.push_back({0,src.shape_.s[i]});
  part.Setup(slice);
  const DAry lsrc=src.Fetch(part_);
  for (int i = 0; i < r.second-r.first; i++) {
    arymath().add(dptr_.get(), dptr_.get(), src.dptr_+i*rowsize, rowsize);
  }
}

/**
  * src must be a matrix
  * this is a vector
  */
void DAry::SumRow(const DAry& src) {
  CHECK_EQ(shape_.dim,1);
  CHECK_EQ(src.shape.dim,2);
  CHECK(src.part_.isLocal());
  int rowlen=shape_.size;
  CHECK_EQ(collen, src.shape_.s[1]);
  for (int i = 0; i < src.shape_.s[0]; i++) {
    arymath().add(dptr_.get(),src.dptr_.get()+rowlen*i, dptr_.get(), rowlen);
  }
}
void DAry::AddRow( const DAry& src) {
  // only support dim being the last dimension
  CHECK_EQ(src.shape.dim,1);
  CHECK(src.part_.isLocal());
  CHECK_EQ(shape_.dim,2);
  int rowlen=src.shape_.size;
  CHECK_EQ(rowlen, shape_.s[1]);
  for (int i = 0; i < shape_.s[0]; i++) {
    arymath().add(dptr_.get()+rowlen*i, dptr_.get()+rowlen*i, src.dptr_.get(), rowlen);
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
  for (int i = 0; i < collen_; i++) {
    dptr[i]+=arymath().sum(srcdptr+i*rowlen, rowlen);
  }
}
// assume both are cached
void DAry::AddCol(const DAry& src) {
  CHECK(src.Cached());
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
  return arymath().sum(dptr_, shape_.size);
}
/**
  * put max(src,x) into dst
  */
void DAry::Max( const DAry& src, float x) {
  Map([x](float v) {return v<x?x:v;}, src);
  PutIfGlobal();
}

/**
  * max element
  */
float DAry::Max(){
  const DAry lary=ga_->Fetch(shape_.ToSlice());
  return arymath().max(lary.dptr_.get(), shape_.size);
}

void DAry::SetLocal(const float x){
  arymath().fill(dptr_.get(), x, part_.size);
}
void DAry::Set(float x){
  ga_.Fill(x, shape_.ToSlice());
}

void DAry::Allocate() {
  if(dptr_!=nullptr){
    LOG(WARNING)<<"the dary has been allocated before, size: "<<alloc_size_;
    delete dptr_;
  }
  if(part_.size==0)
    part_.Size();
  // it is possible the range is empry on some dimension
  dptr_=std::make_shared(new float[part_.size]);
  alloc_size_=part_.size;
}

DAry::~DAry() {
}
}  // namespace lapis
