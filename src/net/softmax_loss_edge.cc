// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-24 15:06
#include <glog/logging.h>
#include <cfloat>
#include "utils/common.h"
#include "net/softmax_loss_edge.h"
namespace lapis {
void SoftmaxLossEdge::Setup(const char flag) {
  Blob &b = bottom_->feature(this);
  num_ = b.num();
  dim_ = b.length() / num_;
}

void SoftmaxLossEdge::Forward(const Blob &src, Blob *dest, bool overwrite) {
  timer.reset();
  VLOG(3)<<"forward softmax loss";
  float *data = src.dptr;
  float *prob = dest->dptr;
  for (int i = 0; i < num_; i++) {
    float mmax = data[0];
    float sum = 0.0f;
    for (int j = 1; j < dim_; j++)
      if (mmax < data[j]) mmax = data[j];
    for (int j = 0; j < dim_; j++) {
      prob[j] = std::exp(data[j] - mmax);
      sum += prob[j];
    }
    for (int j = 0; j < dim_; j++)
      prob[j] /= sum;
    data += dim_;
    prob += dim_;
  }
  CHECK_EQ(data-src.dptr, src.length());
  CHECK_EQ(prob-dest->dptr, dest->length());
  //VLOG(1)<<dest->Norm();
  forward_time_+=timer.elapsed();
}

void SoftmaxLossEdge::Backward(const Blob &src_fea, const Blob &src_grad,
                               const Blob &dest_fea, Blob *dest_grad,
                               bool overwirte) {
  timer.reset();
  VLOG(3)<<"backward softmax loss";
  float *dest = dest_grad->dptr;
  const float *label = src_grad.dptr;
  const float *prob = src_fea.dptr;
  for(int i=0;i<src_fea.length();i++)
    dest[i]=prob[i];
  for (int i = 0; i < num_; i++) {
    int k = static_cast<int>(label[i]);
    dest[i * dim_ + k] -= 1.f;
  }
  for(int i=0;i<src_fea.length();i++)
    dest[i]/=num_;
  //VLOG(1)<<dest_grad->Norm();
  backward_time_+=timer.elapsed();
}

void SoftmaxLossEdge::SetupTopBlob(bool alloc, Blob* blob) {
  blob->Resize(num_, 1, 1, dim_ , alloc);
}
} // namespace lapis
