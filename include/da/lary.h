 Copyright © 2014 Wei Wang. All Rights Reserved.
 2014-10-17 16:36
#ifndef INCLUDE_DA_LARY_H_
#define INCLUDE_DA_LARY_H_
#include "ary.h"

namespace lapis {

class GAry:public Ary{
 public:
  ~GAry();
  GAry():Ary(){}
  void Destroy();
  /**
    * init based on the shape, alloc memory
    */
  void Setup(const Shape& shape, Partition* part);
    /**
    * subdary on the 0-th dim
    */
  GAry Sub(int offset, int size) const ;
  /**
    * Dot production
    */
  void Dot( const GAry& src1, const GAry& src2, bool trans1=false, bool trans2=false);
  void Mult( const GAry& src1, const GAry& src2);
  void Div( const GAry& src1, const GAry& x);
  void Set(float x);
  /**
    * dst=src1+src2
    */
  void Add( const GAry& src1, const GAry& src2);
  void Copy( const GAry& src);
 private:
  void LocateIndex(int* ret, int* orig, int offset);
  int handle_;
  int lo[4],hi[4],ld[3];
};
void GAry::Setup(const Shape& shape, Partition* part){
  CHECK(part.mode==-2);
  SetShape(shape);
  int* chunks=new int[dim_];
  for (int i = 0; i < dim_; i++) {
    chunks[i]=shape_.s[i];
  }
  chunks[part.pdim]=0;
  handle_=NGA_Create(C_FLOAT, dim_, dims, "ga", chunks);
  GA_Set_pgroup(handle_, GlobalContext::Get()->gagroup());
  NGA_Distribution(handle_, GlobalContext::Get()->rank() , lo_, hi_);
  part->lo[0]=lo_[0];
  part->lo[1]=lo_[1];
  part->lo[2]=lo_[2];
  part->lo[3]=lo_[3];
  part->hi[0]=hi_[0];
  part->hi[1]=hi_[1];
  part->hi[2]=hi_[2];
  part->hi[3]=hi_[3];
}

void GAry::Setup(const GAry& other, const vector<Range>& slice){
  SetShape(other.shape_);
  CHECK(slice.size()==dim_);
  for (int i = 0; i < slice.size(); i++) {
    if(i>0)
      ld[i-1]=hi[i]-lo[i];
    lo[i]=slice[i].first;
    hi[i]=slice[i].second-1;
  }
}
void GAry::Destroy(){
  GA_Destroy(handle_);
}

void GAry::Get(float* dptr) {
  NGA_Get(handle_, lo_, hi_,ld_);
}

void GAry::Put(float* dptr) {
  NGA_Put(handle_, lo_, hi_, ld_);
}
void GAry::LocateIndex(int* ret, int* orig, int offset) {
  int size=shape_.SubShape.Size();
  for (i = 0; i < shape_.dim-1; i++) {
    ret[i]=orig+offset/size;
    offset=offset%size;
    size/=shape_.s[i+1];
  }
}

inline GAry GAry::Sub(int offset, int size) const {
  GAry ret;
  ret.SetShape(shape_);
  int* lo=ret.lo_, *hi=ret.hi_, *ld=ret.ld_;
  LocateIndex(lo, lo_, offset);
  LocateIndex(hi, lo_, offset+size);
  bool flag=false;
  for (int i = 0; i < shape_.dim; i++) {
    if(flag) {
      CHECK(hi[i]==0&&lo[i]==0);
      hi[i]=shape_.s[i]-1;
    }else{
      hi[i]=lo[i];
    }
    if(lo[i]!=hi[i]) {
      flag=true;
    }
    if(i>0)
      ld[i-1]=hi[i]-lo[i]+1;
  }
  return ret;
}

// called only when all three GAry are the orignal GAry, no [] operation
void GAry::Dot( const GAry& src1, const GAry& src2, bool trans1=false, bool trans2=false) {
  char transa=trans1?'t':'n';
  char transb=trans2?'t':'n';
  GA_Dgemm(transa, transb, src1.shape(0), src1.shape(1), src2.shape(2), 1.0f,
}

void GAry::Div(const GAry& src1, const GAry& src2) {
  GA_Elem_divide(handle_, src1.handle_, src2.handle_);
}
void GAry::Mult(const GAry& src1, const GAry& src2) {
  GA_Elem_multiply(src1.handle_,src2.handle_, handle_);
}

void GAry::Add(const GAry& src1, const GAry& src2) {
  float a=1.0f, b=1.0f;
  GA_Elem_Add(&a, src1.handle_,&b, src2.handle_, handle_);
}

void GAry::Set(float x) {
  GA_Fill(handle_, &x);
}
}   namespace lapis
#endif   INCLUDE_DA_LARY_H_
