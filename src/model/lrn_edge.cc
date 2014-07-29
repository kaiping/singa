// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-23 22:14

#include "model/lrn_edge.h"
namespace lapis {
void LRNEdge:: Init(const EdgeProto &proto,
                 const std::map<std::string, Layer *> &layer_map) {
  Edge::Init(proto, layer_map);
  local_size_=proto.local_size();
  pre_pad_=(local_size_-1)/2;
  alpha_=proto.alpha();
  beta_=proto.beta();
}

void LRNEdge::Setup(bool set_param) {
  TensorPtr * b=bottom_->feature(this);
  num_=b->num();
  channels_=b->channels();
  height_=b->height();
  width_=b->width();
  pad_square_.Reset(channels_+local_size_-1, height_* width_);
  pad_grad_.Reset(channels_+local_size_-1, height_* width_);
  accum_fea_.Reset(num_, channels_, height_* width_);
}
void LRNEdge::Forward(const Tensor& src, Tensor*dest, bool overwrite) {
  int record_length=channels_*height_*width_;
  int length=height_*width_;
  float alpha_over_size=alpha_/local_size_;
  accum_fea_.setOnes();
  for(int n=0;n<num_;n++) {
    TensorPtr::Square(src.Slice(n),
                      pad_square_.Slice(pre_pad_, pre_pad_+channels_-1)); //ai^2
    Tensor accum_fea_n=accum_fea_.Slice(0);
    Tensor::Sum(pad_square_.Slice(0, local_size),0, accum_fea_n.Slice(0));
    Tensor::MultScalar(accum_fea_n, alpha_over_size, &accum_fea_n); //*alpha
    for (int c=1;c<channels_;c++){
      Tensor accum_fea_n_c=accum_fea_n.Slice(c);
      Tensor::Copy(&accum_fea_n_c, accum_fea_n.Slice(c-1));
      Tensor::Add(pad_square_.Slice(c+local_size_-1), &accum_fea_n_c);
      Tensor::Sub(pad_square_.Slice(c-1), &accum_fea_n_c);
    }
  }
  Tensor::Pow(-beta, accum_fea_, dest);
  Tensor::Mult(src, dest);
}

void LRNEdge::Backward(const Tensor& src_fea, const Tensor& src_grad,
                        const Tensor& dest_fea, Tensor* dest_grad,
                        bool overwrite) {
  int inverse_pre_pad=local_size_-(local_size_+1)/2;
  float factor=-2.*alpha_*beta_/local_size_;
  Tensor::Pow(-beta_, accum_fea_, dest_grad);
  Tensor::Pow(src_grad, dest_grad);
  src_grad.Reshape(num_, channels_, height_*width_);
  src_fea.Reshape(num_, channels_, height_*width_);
  dest_grad->Reshape(num_, channels_, height_*width_);
  Tensor accum_grad(height_*width_);
  for(int n=0;n<num_;n++){
    Tensor accum_fea_n=accum_fea.Slice(n);
    Tensor src_grad_n=src_grad.Slice(n);
    Tensor src_fea_n=src_fea.Slice(n);
    Tensor dest_grad_n=dest_grad.Slice(n);
    Tensor::MultDiv(src_grad_n, src_fea_n, accum_fea_n,
                    pad_grad_.Slice(inverse_pre_pad, inverse_pre_pad+channels_-1));
     //src_grad*b_i/x_i
    // TODO(wangwei) use colwise operation instead of for loop
    Tensor::Sum(pad_grad_.Slice(0, local_size_-1), 0, &accum_grad);
    for (int c=0;c<channels_;c++) {
      Tensor::Add(pad_grad_.Slice(c+local_size_-1), &accum_grad);
      Tensor::Mult(factor, src_fea_n.Slice(c), accum_grad, dest_grad_n.Slice(c));
      Tensor::Sub(pad_grad_.Slice(c), accum_grad);
       // *ai*alpha*beta*2
    }
  }
}

void LRNEdge::SetupTopTensorPtr (TensorPtr t) {
  t->Reshape(num_,channels_,height_, width_);
}

}  // namespace lapis

