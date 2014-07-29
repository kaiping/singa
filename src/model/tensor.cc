// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-16 20:58

#include "model/blob.h"
namespace lapis {

Tensor::Tensor() {
  dim_=0;
  data_=nullptr;
}

Tensor::Tensor(Tensor&& t) {
  dim_=t.dim_;
  shape_[0]=t.shape_[0];
  shape_[1]=t.shape_[1];
  shape_[2]=t.shape_[2];
  shape_[3]=t.shape_[3];
  len_=t.len_;
  len_[0]=t.len_[0];
  len_[1]=t.len_[1];
  len_[2]=t.len_[2];
  len_[3]=t.len_[3];
  data_=t.data_;
}

Tensor& Tensor::opertor=(Tensor&& t) {
  dim_=t.dim_;
  shape_[0]=t.shape_[0];
  shape_[1]=t.shape_[1];
  shape_[2]=t.shape_[2];
  shape_[3]=t.shape_[3];
  len_=t.len_;
  len_[0]=t.len_[0];
  len_[1]=t.len_[1];
  len_[2]=t.len_[2];
  len_[3]=t.len_[3];
  data_=t.data_;
  return *this;
}

Tensor::Tensor(int dim, int* shape, int* len,const shared_ptr<float>& data) {
  dim_=dim;
  for(int i=0;i<dim;i++) {
    shape_[i]=shape[i];
    len_[i]=len[i];
  }
  data_=data;
}

void Tensor::Reshape_(int a, int b, int c) {
  CHECK(a);
  CHECK(b);
  CHECK(c);
  int orig_len_=len_[0];
  dim_=3;
  shape_[0]=a;
  shape_[1]=b;
  shape_[2]=c;
  shape_[3]=0;
  len_[3]=0;
  len_[2]=c;
  len_[1]=b*c;
  len_[0]=len_[1]*a;
  CHECK_EQ(len_[0], orig_len_);
}

void Tensor::Reshape_(int a, int b, int c, int d) {
  CHECK(a);
  CHECK(b);
  CHECK(c);
  CHECK(d);
  int orig_len_=len_gth();
  dim_=4;
  shape_[0]=a;
  shape_[1]=b;
  shape_[2]=c;
  shape_[3]=d;
  len_[3]=d;
  len_[2]=d*c;
  len_[1]=len_[2]*b;
  len_[0]=len_[1]*a;
  CHECK_EQ(len_[0], orig_len_);
}

void Tensor::Reset(int a, int b, int c, int d) {
  int orig_len_=len_gth();
  Reshape_(a,b,c,d);
  if(data_!=nullptr&&orig_len_==len_gth())
    return;
  else
    data_=new float[len_gth()];
}

Tensor Tensor::Slice(int start) {
  Tensor t(dim_-1, shape_,len_, at(start));
  return t;
}

void Tensor::Sum(const Tensor&& src, int aixs, Tensor *dest, bool overwrite=true) {

}

void Tensor::Dot(const Tensor&& A, const Tensor&&B, Tensor *C, bool overwrite=true) {

}

void Tensor::Dot(const Tensor&& A, const Tensor &&B, Tensor &&C,
    bool transA, bool transB, bool overwriteC=true) {

}
void Tensor::Dot(const Tensor& A, const Tensor &B, Tensor *C, bool overwrite=true) {

void Tensor::Add(TensorPtr A, TensorPtr B, TensorPtr C, bool overwrite=true) {

}
}  // namespace lapis
