// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-24 15:06
#include <glog/logging.h>
#include <cfloat>

#include "model/softmax_loss_edge.h"
namespace lapis {
void SoftmaxLossEdge::Setup(bool set_param) {
  Blob* b=bottom_->feature(this);
  prob_.Reshape(b->num(), b->record_length());
}

void SoftmaxLossEdge::Forward(const Blob *src, Blob *dest, bool overwrite) {
  AMat src_mat(src->mutable_data(), src->num(),src->record_length());
  AMat prob_mat(prob_.mutable_data(), prob_.num(),prob_.record_length());
  prob_mat=src_mat.exp();
  prob_mat=prob_mat.colwise()/(prob_mat.rowwise().sum());
}

void SoftmaxLossEdge::Backward(const Blob *src_fea, const Blob *src_grad,
                        const Blob *dest_fea, Blob *dest_grad,
                        bool overwirte) {
  const float* prob_data=prob_.data();
  const float* label=src_fea->data();
  float* dest_grad_data=dest_grad->mutable_data();
  memcpy(dest_grad_data, prob_data, prob_.length()*sizeof(float));
  int record_length=dest_grad->record_length();
  float loss=0;
  for(int i=0;i<dest_grad->num();i++){
    int k=static_cast<int>(label[i]);
    dest_grad_data[i*record_length+k]-=1.f;
    loss+=-log(std::max(prob_data[i*record_length+k], FLT_MIN));
  }
}
}  // namespace lapis
