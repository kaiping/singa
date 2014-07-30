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
  TShape4& shape = (bottom_->feature(this)).shape;
  num_ = shape[3];
  channels_ = shape[2];
  height_ = shape[1];
  width_ = shape[0];
  pad_square_.Resize( Shape2(height_ * width_, channels_ + local_size_ - 1));
  pad_grad_.Resize( Shape2(height_ * width_, channels_ + local_size_ - 1));
  accum_fea_.Resize( Shape3(height_ * width_, channels_, num_), 1.0);
  accum_grad_.Resize( Shape1(height_ * width_));
}
void LRNEdge::Forward(const Blob4 &src, Blob4 *dest, bool overwrite) {
  float alpha_over_size = alpha_ / local_size_;
  Tensor3 src3(src.dptr, Shape3(height_ * width_, channels_, num_));
  Tensor3 dest3(dest->dptr, Shape3(height_ * width_, channels_, num_));
  Tensor2 accum_fea2;
  for (int n = 0; n < num_; n++) {
    accum_fea2 = accum_fea_[n];
    pad_square_.Slice(pre_pad_, pre_pad_ + channels_ - 1) =
      mshadow::expr::F<mshadow::op::square>(src3[n]) * alpha_over_size; //ai^2
    accum_fea2[0] = sum_rows(pad_square_.Slice(0, local_size_));
    for (int c = 1; c < channels_; c++)
      accum_fea2[c] = accum_fea2[c - 1] + pad_square_[c + local_size_ - 1] -
                      pad_square_[c - 1];
  }
  dest3 = mshadow::expr::F<mshadow::op::power>(accum_fea_, -beta_) * src3;
}

void LRNEdge::Backward(const Blob4 &src_fea, const Blob4 &src_grad,
                       const Blob4 &dest_fea, Blob4 *dest_grad,
                       bool overwrite) {
  int inverse_pre_pad = local_size_ - (local_size_ + 1) / 2;
  float factor = -2.*alpha_ * beta_ / local_size_;
  Tensor3 src_fea3(src_fea.dptr, Shape3(height_ * width_, channels_, num_));
  Tensor3 src_grad3(src_grad.dptr, Shape3(height_ * width_, channels_, num_));
  Tensor3 dest_grad3(dest_grad->dptr, Shape3(height_ * width_, channels_, num_));
  dest_grad3 = mshadow::expr::F<mshadow::op::power>(accum_fea_, -beta_) * src_grad3;
  Tensor2 src_fea2, dest_grad2;
  for (int n = 0; n < num_; n++) {
    src_fea2=src_fea3[n];
    pad_grad_.Slice(inverse_pre_pad, inverse_pre_pad + channels_) =
      src_grad3[n] * src_fea3[n] / accum_fea_[n];
    accum_grad_ = mshadow::expr::sum_rows(pad_grad_.Slice(0, local_size_ - 1));
    //src_grad*b_i/x_i
    // TODO(wangwei) use colwise operation instead of for loop
    dest_grad2=dest_grad3[n];
    for (int c = 0; c < channels_; c++) {
      accum_grad_ += pad_grad_[c + local_size_ - 1];
      dest_grad2[c] += src_fea2[c] * accum_grad_ * factor;
      accum_grad_ -= pad_grad_[c];
      // *ai*alpha*beta*2
    }
  }
}

void LRNEdge::SetupTopBlob(Blob4 *blob) {
  blob->Resize(Shape4(width_, height_, channels_, num_));
}

}  // namespace lapis

