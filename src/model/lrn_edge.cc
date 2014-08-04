// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-23 22:14

#include <glog/logging.h>

#include "model/lrn_edge.h"
#include "mshadow/tensor.h"

namespace lapis {
void LRNEdge:: Init(const EdgeProto &proto,
                    const std::map<std::string, Layer *> &layer_map) {
  Edge::Init(proto, layer_map);
  local_size_ = proto.local_size();
  pre_pad_ = (local_size_ - 1) / 2;
  alpha_ = proto.alpha();
  beta_ = proto.beta();
  knorm_=proto.knorm();
}

void LRNEdge::Setup(bool set_param) {
  Blob &b = bottom_->feature(this);
  num_ = b.num();
  channels_ = b.channels();
  height_ = b.height();
  width_ = b.width();
  pad_tmp_.Resize(1,channels_ + local_size_ - 1,1, height_ * width_);
  accum_fea_.Resize( num_, channels_,1, height_ * width_);
  accum_grad_.Resize(1,1,1,height_ * width_);

  VLOG(2)<<"padded image shape "<<pad_tmp_.tostring();
  VLOG(2)<<"accum fea shape "<<accum_fea_.tostring();
  VLOG(2)<<"accum grad shape "<<accum_grad_.tostring();
}
void LRNEdge::Forward(const Blob &src, Blob *dest, bool overwrite) {
  VLOG(3)<<name_;
  float alpha_over_size = alpha_ / local_size_;
  Tensor2 pad_square(pad_tmp_.dptr,
                     Shape2(channels_ + local_size_ - 1,height_ * width_));
  Tensor3 accum_fea3(accum_fea_.dptr,
                    Shape3( num_, channels_,height_ * width_));
  accum_fea3 = knorm_;
  Tensor3 src3(src.dptr, Shape3(num_, channels_, height_ * width_));
  for (int n = 0; n < num_; n++) {
    pad_square.Slice(pre_pad_, pre_pad_ + channels_) =
      mshadow::expr::F<mshadow::op::square>(src3[n]) * alpha_over_size; //ai^2
    Tensor2 accum_fea2= accum_fea3[n];
    accum_fea2[0] += sum_rows(pad_square.Slice(0, local_size_));
    for (int c = 1; c < channels_; c++)
      accum_fea2[c] += accum_fea2[c - 1] + pad_square[c + local_size_ - 1] -
                      pad_square[c - 1];
  }
  Tensor3 dest3(dest->dptr, Shape3(num_, channels_, height_ * width_));
  dest3 = mshadow::expr::F<mshadow::op::power>(accum_fea3, -beta_) * src3;
}

void LRNEdge::Backward(const Blob &src_fea, const Blob &src_grad,
                       const Blob &dest_fea, Blob *dest_grad,
                       bool overwrite) {
  VLOG(3)<<name_;
  int inverse_pre_pad = local_size_ - (local_size_ + 1) / 2;
  float factor = -2.*alpha_ * beta_ / local_size_;

  Tensor3 accum_fea(accum_fea_.dptr, Shape3(num_, channels_,height_ * width_));
  Tensor3 src_grad3(src_grad.dptr, Shape3(num_,channels_,height_ * width_));
  Tensor3 dest_grad3(dest_grad->dptr, Shape3(num_, channels_, height_ * width_));
  dest_grad3 = mshadow::expr::F<mshadow::op::power>(accum_fea, -beta_)
    * src_grad3; // the first part, 1/x_i^beta

  Tensor1 accum_grad(accum_grad_.dptr, Shape1(height_ * width_));
  Tensor2 pad_grad(pad_tmp_.dptr,
                   Shape2( channels_ + local_size_ - 1,height_ * width_));
  Tensor3 src_fea3(src_fea.dptr, Shape3(num_, channels_, height_ * width_));
  for (int n = 0; n < num_; n++) {
    Tensor2 src_fea2 = src_fea3[n];
    pad_grad.Slice(inverse_pre_pad, inverse_pre_pad + channels_) =
      src_grad3[n] * src_fea3[n] / accum_fea[n]; //src_grad*b_i/x_i
    accum_grad = mshadow::expr::sum_rows(pad_grad.Slice(0, local_size_ - 1));
    // TODO(wangwei) use colwise operation instead of for loop
    Tensor2 dest_grad2 = dest_grad3[n];
    for (int c = 0; c < channels_; c++) {
      accum_grad += pad_grad[c + local_size_ - 1];
      // +src_grad*factor*b_i*a_j/x_i
      dest_grad2[c] += src_fea2[c] * accum_grad * factor;
      accum_grad -= pad_grad[c];
      // *ai*alpha*beta*2
    }
  }
}

void LRNEdge::SetupTopBlob(Blob *blob) {
  blob->Resize(num_, channels_, height_, width_);
}

}  // namespace lapis

