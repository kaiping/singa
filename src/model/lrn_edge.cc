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
  Blob4Ptr * b=bottom_->feature(this);
  num_=b->num();
  channels_=b->channels();
  height_=b->height();
  width_=b->width();
  pad_square_.Resize( Shape2(height_* width_,channels_+local_size_-1));
  pad_grad_.Resize( Shape2(height_* width_, channels_+local_size_-1));
  accum_fea_.Resize( Shape3(height_* width_,channels_, num_), 1.0);
  accum_grad_.Resize( Shape1(height_* width_));
}
void LRNEdge::Forward(const Blob4& src, Blob4*dest, bool overwrite) {
  float alpha_over_size=alpha_/local_size_;
  Blob3 src3=reshape(src, Shape3(height_*width_, channels_, num_));
  Blob3 dest3=reshape(*dest, Shape3(height_*width_, channels_, num_));
  for(int n=0;n<num_;n++) {
    Blob2 accum_fea2=accum_fea_[n];
    pad_square_.Slice(pre_pad_, pre_pad_+channels_-1)=F<op::square>(src3[n]) *
                                                      alpha_over_size; //ai^2
    accum_fea2[0]=sum_rows(pad_square_.Slice(0, local_size_));

    for (int c=1;c<channels_;c++)
      accum_fea2[c]=accum_fea2[c-1]+pad_square_[c+local_size_-1]-pad_square_[c-1];
  }
  dest3=F<op:power>(accum_fea, -beta)*src3;
}

void LRNEdge::Backward(const Blob4& src_fea, const Blob4& src_grad,
                        const Blob4& dest_fea, Blob4* dest_grad,
                        bool overwrite) {
  int inverse_pre_pad=local_size_-(local_size_+1)/2;
  float factor=-2.*alpha_*beta_/local_size_;

  Blob3 src_fea3=reshape(src_fea, Shape3(height_*width_, channels_, num_));
  Blob3 src_grad3=reshape(src_grad, Shape3(height_*width_, channels_, num_));
  Blob3 dest_grad3=reshape(*dest_grad, Shape3(height_*width_, channels_, num_));
  dest_grad3=F<op:power>(accum_fea_, -beta)*src_grad3;
  for(int n=0;n<num_;n++){
    Blob2 src_fea2=src_fea3[n];
    pad_grad_.Slice(inverse_pre_pad_, inverse_pre_pad_+channels_)=
      src_grad[n]*src_fea2/accum_fea_[n];
    accum_grad=summ_rows(pad_grad_.Slice(0, local_size_-1));

     //src_grad*b_i/x_i
    // TODO(wangwei) use colwise operation instead of for loop
    for (int c=0;c<channels_;c++) {
      accum_grad+=pad_grad_[c+local_size_-1];
      dest_grad3[n]+=src_fea2[c]*accum_grad*factor;
      accum_grad-=pad_grad_[c];
       // *ai*alpha*beta*2
    }
  }
}

void LRNEdge::SetupTopBlob (Blob4 *t) {
  t->Resize(Shape4(width_,height_,channels_, num_));
}

}  // namespace lapis

