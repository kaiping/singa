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
  Blob* b=bottom_->feature(this);
  num_=b->num();
  channels_=b->channels();
  height_=b->height();
  width_=b->width();
  pad_square_.Reshape(1, channels_+local_size_-1, height_, width_);
  pad_grad_.Reshape(1, channels_+local_size_-1, height_, width_);
  accum_fea_.Reshape(num_, channels_, height_, width_);
}
void LRNEdge::Forward(const Blob *src, Blob *dest, bool overwrite) {
  int record_length=channels_*height_*width_;
  int length=height_*width_;
  float alpha_over_size=alpha_/local_size_;
  AVec pad_square_vec(pad_square_.mutable_data(), pad_square_.length());
  AVec accum_fea_vec(accum_fea_.mutable_data(), accum_fea_.length());
  accum_fea_vec.setOnes();
  AVec src_fea_vec(src->offset(0), record_length);
  AMat pad_square_mat(pad_square_.mutable_data(), local_size_, length);;
  for(int n=0;n<num_;n++) {
    new (&pad_square_vec)AVec(pad_square_.offset(0, pre_pad_), record_length);
    new (&src_fea_vec)AVec(src->offset(n), record_length);
    pad_square_vec=src_fea_vec.square(); //ai^2

    new (&accum_fea_vec)AVec(accum_fea_.offset(n),length);
    accum_fea_vec+=pad_square_mat.colwise().sum()*alpha_over_size; //*alpha

    for (int c=1;c<channels_;c++){
      memcpy(accum_fea_.offset(n,c), accum_fea_.offset(n,c-1), length*sizeof(float));
      new (&pad_square_vec)AVec(pad_square_.offset(0, c+local_size_-1), length);
      new (&accum_fea_vec)AVec(accum_fea_.offset(n,c), length);
      accum_fea_vec+=pad_square_vec*alpha_over_size;
      new (&pad_square_vec)AVec(pad_square_.offset(0,c-1), length);
      accum_fea_vec-=pad_square_vec*alpha_over_size;
    }
  }
  new (&accum_fea_vec)AVec(accum_fea_.mutable_data(), record_length);
  new (&src_fea_vec)AVec(src->mutable_data(), record_length);
  AVec dest_fea_vec(dest->mutable_data(), record_length);
  dest_fea_vec=accum_fea_vec.pow(-beta_)*src_fea_vec; //ai*xi^(-beta)
}

void LRNEdge::Backward(const Blob *src_fea, const Blob *src_grad,
                        const Blob *dest_fea, Blob *dest_grad,
                        bool overwrite) {
  int inverse_pre_pad=local_size_-(local_size_+1)/2;
  int record_length=channels_*height_*width_;
  int length=height_*width_;
  float factor=2.*alpha_*beta_/local_size_;

  AVec src_grad_vec(src_grad->mutable_data(), src_grad->length());
  AVec dest_grad_vec(dest_grad->mutable_data(), dest_grad->length());
  AVec accum_fea_vec(accum_fea_.mutable_data(), accum_fea_.length());
  dest_grad_vec=accum_fea_vec.pow(-beta_)*src_grad_vec;
  AVec dest_fea_vec(dest_fea->mutable_data(), dest_fea->length());
  EigenAVector grad_vec(length);
  AVec pad_grad_vec(pad_grad_.offset(0, inverse_pre_pad), record_length);
  AVec src_fea_vec(src_fea->offset(0), record_length);
  AMat pad_grad_mat(pad_grad_.mutable_data(), local_size_-1, length);
  for(int n=0;n<num_;n++){
    new (&pad_grad_vec)AVec(pad_grad_.offset(0, inverse_pre_pad), record_length);
    new (&src_fea_vec)AVec(src_fea->offset(n), record_length);
    new (&src_grad_vec)AVec(src_grad->offset(n), record_length);
    new (&accum_fea_vec)AVec(accum_fea_.offset(n), record_length);
    pad_grad_vec=src_grad_vec*src_fea_vec/accum_fea_vec; //src_grad*b_i/x_i
    // TODO(wangwei) use colwise operation instead of for loop
    grad_vec=pad_grad_mat.colwise().sum();
    for (int c=0;c<channels_;c++) {
      new (&pad_grad_vec)AVec(pad_grad_.offset(0,c+local_size_-1), length);
      grad_vec+=pad_grad_vec;
      new (&dest_fea_vec)AVec(dest_fea->offset(n,c), length);
      new (&dest_grad_vec)AVec(dest_grad->offset(n,c), length);
      dest_grad_vec+=grad_vec*dest_fea_vec*factor; // *ai*alpha*beta*2
      new (&pad_grad_vec)AVec(pad_grad_.offset(0,c), length);
      grad_vec-=pad_grad_vec;
    }
  }
}

void LRNEdge::SetupTopBlob(Blob* blob) {
  blob->Reshape(num_,channels_,height_, width_);
}

}  // namespace lapis

