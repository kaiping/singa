// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-23 22:14

#include "model/lrn_edge.h"
#include "mshadow/tensor_expr.h"
#include "mshadow/tensor_base.h"
#include "mshadow/tensor_expr_ext.h"

namespace lapis {
void LRNEdge:: Init(const EdgeProto &proto,
                    const std::map<std::string, Layer *> &layer_map) {
  Edge::Init(proto, layer_map);
  local_size_ = proto.local_size();
  pre_pad_ = (local_size_ - 1) / 2;
  alpha_ = proto.alpha();
  beta_ = proto.beta();
}

void LRNEdge::Setup(bool set_param) {
  Blob &b = bottom_->feature(this);
  num_ = b.num();
  channels_ = b.channels();
  height_ = b.height();
  width_ = b.width();
  pad_square_.Resize(height_ * width_, channels_ + local_size_ - 1);
  pad_grad_.Resize(height_ * width_, channels_ + local_size_ - 1);
  accum_fea_.Resize(height_ * width_, channels_, num_);
  accum_grad_.Resize(height_ * width_);
}
void LRNEdge::Forward(const Blob &src, Blob *dest, bool overwrite) {
  float alpha_over_size = alpha_ / local_size_;
  Tensor2 pad_square(pad_square_.dptr, Shape2(height_ * width_,
                     channels_ + local_size_ - 1));
  Tensor2 pad_grad(pad_grad_.dptr, Shape2(height_ * width_,
                                          channels_ + local_size_ - 1));
  Tensor3 accum_fea(accum_fea_.dptr, Shape3(height_ * width_, channels_, num_));
  accum_fea = 1.0;
  Tensor3 src3(src.dptr, Shape3(height_ * width_, channels_, num_));
  Tensor3 dest3(dest->dptr, Shape3(height_ * width_, channels_, num_));
  Tensor2 accum_fea2;
  for (int n = 0; n < num_; n++) {
    accum_fea2 = accum_fea[n];
    pad_square.Slice(pre_pad_, pre_pad_ + channels_ - 1) =
      mshadow::expr::F<mshadow::op::square>(src3[n]) * alpha_over_size; //ai^2
    accum_fea2[0] = sum_rows(pad_square.Slice(0, local_size_));
    for (int c = 1; c < channels_; c++)
      accum_fea2[c] = accum_fea2[c - 1] + pad_square[c + local_size_ - 1] -
                      pad_square[c - 1];
  }
  dest3 = mshadow::expr::F<mshadow::op::power>(accum_fea, -beta_) * src3;
}

void LRNEdge::Backward(const Blob &src_fea, const Blob &src_grad,
                       const Blob &dest_fea, Blob *dest_grad,
                       bool overwrite) {
  int inverse_pre_pad = local_size_ - (local_size_ + 1) / 2;
  float factor = -2.*alpha_ * beta_ / local_size_;
  Tensor2 pad_square(pad_square_.dptr, Shape2(height_ * width_,
                     channels_ + local_size_ - 1));
  Tensor2 pad_grad(pad_grad_.dptr, Shape2(height_ * width_,
                                          channels_ + local_size_ - 1));
  Tensor3 accum_fea(accum_fea_.dptr, Shape3(height_ * width_, channels_, num_));
  Tensor1 accum_grad(accum_grad_.dptr, Shape1(height_ * width_));
  Tensor3 src_fea3(src_fea.dptr, Shape3(height_ * width_, channels_, num_));
  Tensor3 src_grad3(src_grad.dptr, Shape3(height_ * width_, channels_, num_));
  Tensor3 dest_grad3(dest_grad->dptr, Shape3(height_ * width_, channels_, num_));
  dest_grad3 = mshadow::expr::F<mshadow::op::power>(accum_fea,
               -beta_) * src_grad3;
  Tensor2 src_fea2, dest_grad2;
  for (int n = 0; n < num_; n++) {
    src_fea2 = src_fea3[n];
    pad_grad.Slice(inverse_pre_pad, inverse_pre_pad + channels_) =
      src_grad3[n] * src_fea3[n] / accum_fea[n];
    accum_grad = mshadow::expr::sum_rows(pad_grad.Slice(0, local_size_ - 1));
    //src_grad*b_i/x_i
    // TODO(wangwei) use colwise operation instead of for loop
    dest_grad2 = dest_grad3[n];
    for (int c = 0; c < channels_; c++) {
      accum_grad += pad_grad[c + local_size_ - 1];
      dest_grad2[c] += src_fea2[c] * accum_grad * factor;
      accum_grad -= pad_grad[c];
      // *ai*alpha*beta*2
    }
  }
}

void LRNEdge::SetupTopBlob(Blob *blob) {
  blob->Resize(width_, height_, channels_, num_);
}

}  // namespace lapis

