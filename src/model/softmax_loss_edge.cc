// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-24 15:06
#include <glog/logging.h>
#include <cfloat>

#include "model/softmax_loss_edge.h"
namespace lapis {
void SoftmaxLossEdge::Setup(bool set_param) {
  Blob &b = bottom_->feature(this);
  num_ = b.num();
  dim_ = b.length() / num_;
  prob_.Resize(num_, 1,1 ,dim_);
  VLOG(2)<<"prob shape "<<prob_.tostring();
}

void SoftmaxLossEdge::Forward(const Blob &src, Blob *dest, bool overwrite) {
  VLOG(3)<<name_;
  float *data = src.dptr;
  float *prob = prob_.dptr;
  for (int i = 0; i < num_; i++) {
    float mmax = data[0];
    float sum = 0.0f;
    for (int j = 0; j < dim_; j++)
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
}

void SoftmaxLossEdge::Backward(const Blob &src_fea, const Blob &src_grad,
                               const Blob &dest_fea, Blob *dest_grad,
                               bool overwirte) {
  VLOG(3)<<name_;
  float *dest = dest_grad->dptr;
  const float *label = src_fea.dptr;
  const float *prob = prob_.dptr;
  float loss = 0;
  for(int i=0;i<prob_.length();i++)
    dest[i]=prob[i];
  for (int i = 0; i < num_; i++) {
    int k = static_cast<int>(label[i]);
    dest[i * dim_ + k] -= 1.f;
    loss += -log(std::max(prob[i * dim_ + k], FLT_MIN));
  }
  for(int i=0;i<prob_.length();i++)
    dest[i]/=num_;
}
}  // namespace lapis
