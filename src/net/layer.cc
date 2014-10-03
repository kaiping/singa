// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-06 15:19

#include <glog/logging.h>
#include <memory>
#include <random>
#include "mshadow/tensor.h"


#include "net/layer.h"
#include "net/data_layer.h"
#include "net/linear_layer.h"
#include "net/relu_layer.h"

namespace lapis {
/*****************************************************************************
 * Implementation for Layer
 *****************************************************************************/
void Layer::Init(const LayerProto &proto) {
  name_ = proto.name();
  type_=proto.type();
  if(proto.has_data()){
    data_.InitFromProto(proto.data());
    grad_.InitFromProto(proto.data());
  }
}

void Layer::ToProto(LayerProto *proto, bool copyData) {
  proto->set_name(name_);
  proto->set_type(type_);
  DAryProto *data=proto->set_data();
  data_.ToProto(data, copyData);
  DAryProto *grad=proto->set_grad();
  grad_.ToProto(grad, copyData);
}

void Layer::SetupDAryShape(const vector<vector<int>>& shapes){ }
void Layer::SetupDAryShape(){}
void Layer::AllocMemory(){}
void Layer::ComputeFeature(){}
void Layer::ComputeGradient(){}
void Layer::CollectParams(vector<Param*> *params){}

/*****************************************************************************
 * Implementation for ConvLayer
 *****************************************************************************/
void ConvLayer::Init(const LayerProto &proto) {
  Layer::Init(proto);
  CHECK(proto.has_kernel_size());
  ksize_ = proto.kernel_size();
  stride_ = proto.stride();
  pad_ = proto.pad();
  nkernels_ = proto.num_output();
  ngroups_ = proto.num_groups();

}
void Layer::CollectParams(vector<Param*> *params){
  weight_.set_id(params->size());
  params->push_back(&weight_);
  bias_.set_id(params->size());
  params->push_back(&bias_);
}
void ConvLayer::ToProto(LayerProto *proto, bool copyData) {
  Layer::ToProto(proto, copyData)
  proto->set_kernel_size(ksize_);
  proto->set_stride(stride_);
  proto->set_pad(pad_);
  proto->set_num_kernels(nkernels_);
  proto->set_num_groups(ngroups_);
  ParamProto* weight=proto->set_weight();
  weight_.ToProto(weight, copyData);
  ParamProto* bias=proto->set_bias();
  bias_.ToProto(bias, copyData);
}

void ConvLayer::SetupDAryShape(){
  CHECK_EQ(in_edges_.size(), 1);
  CHECK_EQ(out_edges_.size(), 1);
  const DAry& bottom=in_edges_[0]->GetData(this);
  num_ = bottom.Shape(0);
  channels_ = bottom.Shape(1);
  height_ = bottom.Shape(2);
  width_ = bottom.Shape(3);
  // height and width of the image after convolution
  conv_height_ = (height_ + 2 * pad_ - ksize_) / stride_ + 1;
  conv_width_ = (height_ + 2 * pad_ - ksize_) / stride_ + 1;
  vector<int> shape{num_, nkernels_, conv_height_, conv_width_};
  data_.SetShape(shape);
  grad_.SetShape(shape);
  weight_.SetShape(nkernels_, ksize_*ksize_*channels_);
  bias_.SetShape(nkernels_);
  // weight matrix is of size nkernels_* K_, col_fea is of size
  // num_groups*K_*N_, image after conv is of shape (num_kernels_*N_)
  CHECK_EQ(nkernels_ % num_groups_ , 0)<< nkernels_<<", "<<ngroups_;
  CHECK_EQ((ksize_ * ksize_ * channels_) % ngroups_, 0)<<ksize<<":"<<channels_<<":"<<ngroups_;
  M_ = nkernels_ / num_groups_;
  K_ = ksize_ * ksize_ * channels_ / num_groups_;
  N_ = conv_height_ * conv_width_;
  col_data_.SetShape(num_, num_groups_, K_, N_);
  col_grad_.SetShape(num_, num_groups_, K_, N_);
}

void ConvLayer::AllocMemory(){
  data_.AllocMemory();
  grad_.AllocMemory();
  weight_.AllocMemory();
  bias_.AllocMemory();
}

void ConvLayer::ComputeFeature() {
  VLOG(3)<<name_;
  const DAry& bottom=in_edges_[0]->GetData(this);
  Img2Col(&col_data_, bottom);
  // copy constructor with rvalue ref DAry(DAry&& other)
  DAry data4(data_,{num_, ngroups_, K_, N_});
  DAry bottom4(bottom,{num_, ngroups_, M_, N_});
  DAry weight3(weight_.data(),{ngroups_, M_, K_});
  for (int n = 0; n < num_; n++) {
    // copy constructor with rvalue ref
    DAry data3=data4[n];
    DAry bottom3=bottom[n];
    for (int g = 0; g < ngroups_; g++)
      DAry::Dot(data3[g], weight3[g], bottom3[g]);
  }
  DAry data3(data_,{num_, nkernels_,N_});
  // add bias to data3 along the 1-th dim (start at 0-th dim)
  DAry::AddVec(&data3, bias_.data(), 1);
}

void ConvLayer::ComputeGradient() {
  VLOG(3)<<name_;
  {
    DAry *gbias=bias_.mutable_grad();
    DAry grad3(grad_, {num_, nkernels_, N_});
    // sum along 1-th dim, i.e., the result aray has length as the 1-th dim
    SumExcept(gbias, grad3, 1);
  }

  const DAry weight3(weight_.data(), {ngroups_, M_, K_});
  DAry gweight3(*weight_.mutable_grad(), {ngroups_, M_, K_});
  DAry col4(col_data_, {num_, ngroups_,K_, N_});
  DAry* gbottom=in_edges_[0]->GetMutableGrad(this);
  if(gbottom!=nullptr){
    DAry gcol4(col_grad_, {num_, ngroups_,K_, N_});
    for (int n = 0; n < num_; n++) {
      DAry grad3(graid[n], {ngroups_,M_,N_});
      DAry col3=col4[n];
      DAry gcol3=gcol4[n];
      for (int g = 0; g < num_groups_; g++) {
        Dot(gweight3[g], grad3[g], col3[g], trans1=false, trans2=true);
        Dot(gcol3[g], weight3[g], grad3[g], trans1=true, trans2=false);
      }
    }
    Col2Img(gbottom, col_grad_, window_, pad_, stride_);
  } else {
    for (int n = 0; n < num_; n++) {
      DAry grad3(grad[n], {ngroups_,M_,N_});
      DAry col3=col4[n];
      DAry gcol3=gcol4[n];
      for (int g = 0; g < num_groups_; g++)
        Dot(gweight3[g], grad3[g], col3[g], trans1=false, trans2=true);
    }
  }
}

/*
 * only consider parition along the num dimension
 */
void ConvEdge::Img2Col(DAry* dst, const DAry& src){
  Range nrng=dst->IdxRng(0);
  std::vector<Range> slice{nrng, {0, channels_*ksize_*ksize_},
    {0, cheight_}, {0, cwidth_}};
  const DAry lsrc=src.FetchData(slice);
  for(int n=nrng.first; n<nrng.second;n++){
    for (int c = 0; c < channels_; ++c) {
      int w_offset = c % ksize;
      int h_offset = (c / ksize) % ksize;
      int c_im = c / ksize / ksize;
      for (int h = 0; h < cheight_; ++h) {
        for (int w = 0; w < cwidth_; ++w) {
          int h_pad = h * stride - pad + h_offset;
          int w_pad = w * stride - pad + w_offset;
          if (h_pad >= 0 && h_pad < height_ && w_pad >= 0 && w_pad < width_)
            dst->at(n,c,h,w)=lsrc.get(n,c,h_pad, w_pad);
          else
            dst->at(n,c,h,w)= 0;
        }
      }
    }
  }
}
/*
 * consider only partition on num dimension
 */
void ConvEdge::Col2Img(DAry* dst, const DAry& src){
  Range nrng=dst->IdxRng(0);
  const DAry lsrc=src.FetchData(slice);
  lsrc.set(0);
  for(int n=nrng.first;n<nrng.second;n++){
    for (int c = 0; c < channels_; ++c) {
      int w_offset = c % ksize;
      int h_offset = (c / ksize) % ksize;
      int c_im = c / ksize / ksize;
      for (int h = 0; h < cheight_; ++h) {
        for (int w = 0; w < cwidth_; ++w) {
          int h_pad = h * stride - pad + h_offset;
          int w_pad = w * stride - pad + w_offset;
          if (h_pad >= 0 && h_pad < height_ && w_pad >= 0 && w_pad < width_)
            dst->at(n,c,h_pad, w_pad) += lsrc.get(n,c,h,w);
        }
      }
    }
  }
}

/*****************************************************************************
 * Implementation for ReLULayer
 *****************************************************************************/
void ReLULayer::SetupDAryShape(){
  CHECK_EQ(in_edges_.size(),1);
  const DAry& bottom=in_edges_[0]->GetData(this);
  data_.SetShape(bottom.Shape());
  grad_.SetShape(bottom.Shape());
}

void ReLULayer::AllocMemory(){
  data_.AllocMemory();
  grad_.AllocMemory();
}

void ReLULayer::ComputeFeature() {
  DAry::Max(&data_, in_edges_[0].GetData(this), 0);
}

void ReLULayer::ComputeGradient() {
  DAry* gbottom=in_edges_[0]->GetMutableGrad(this);
  const DAry& bottom=in_edges_[0]->GetData(this);
  DAry::Map(gbottom, [](float d, float g){return d>0?g:0;}, bottom, grad_);
}
/*****************************************************************************
 * Implementation for DropoutLayer
 *****************************************************************************/
void DropoutLayer::Init(const LayerProto &proto) {
  Layer::Init(proto);
  drop_prob_=proto.drop_prob();
  if(proto_.has_data());
    mask_.InitLike(data_);
}

void DropoutLayer::ToProto(LayerProto *proto, bool copyData) {
  Layer::ToProto(proto, copyData);
  proto->set_drop_prob(drop_prob_);
}

void DropoutLayer::SetupDAryShape(){
  CHECK_EQ(in_edges_.size(),1);
  const DAry& bottom=in_edges_[0]->GetData(this);
  data_.SetShape(bottom.Shape());
  grad_.SetShape(bottom.Shape());
  mask_.SetShape(bottom.Shape());
}

void DropoutLayer::AllocMemory(){
  data_.AllocMemory();
  grad_.AllocMemory();
  mask_.AllocMemory();
}
void DropoutLayer::ComputeFeature() {
  float keep_prob = 1.0 - drop_prob_;
  mask_.Random();
  DAry::Threshold(mask_, mask_, keep_prob);
  //DAry::Map(&mask_, [keep_prob](float v){return v<=keep_prob?1.0f:0.0f;}, mask_);
  float scale=1.0/keep_prob;
  const DAry& bottom=in_edges_[0]->GetData(this);
  DAry::Map(&data_, [scale](float v, float m) {return v*m*scale;}, bottom, mask_);
}

void DropoutLayer::ComputeGradient() {
  DAry* gbottom=in_edges_[0]->GetMutableGrad(this);
  DAry::Mult(gbottom, grad_, mask_);
}
/*****************************************************************************
 * Implementation for PoolingLayer
 *****************************************************************************/
void PoolingLayer::Init(const LayerProto &proto) {
  wsize_ = proto.window_size();
  stride_ = proto.stride();
  pooling_method_ = proto.pooling_method();
}
void DropoutLayer::ToProto(LayerProto *proto, bool copyData) {
  Layer::ToProto(proto, copyData);
  proto->set_window_size(wsize_);
  proto->set_stride(stride_);
  proto->set_pooling_method(pooling_method_);
}
void PoolingLayer::SetupDAryShape(){
  const DAry& bottom=in_edges_[0]->GetData(this);
  num_ = bottom.Shape(0);
  channels_ = bottom.Shape(1);
  height_ = bottom.Shape(2);
  width_ = bottom.Shape(3);
  pheight_ = static_cast<int> (
                   ceil(static_cast<float>(height_ - wsize_) / stride_)) + 1;
  pwidth_ = static_cast<int> (
                  ceil(static_cast<float>(width_ - wsize_) / stride_)) + 1;
  vector<int> shape{num_, channels_, pheight_, pwidth_};
  data_.SetShape(shape);
  grad_.SetShape(shape);
}

void PoolingLayer::AllocMemory(){
  data_.AllocMemory();
  grad_.AllocMemory();
}
void PoolingLayer::ComputeFeature() {
  Range nrng=data_.IndexRange(0);
  vector<Range> slice{nrng, {0, channels_},{0, pheight_},{0,pwidth_}};
  const DAry& bottom=in_edges_[0]->GetData(this);
  DAry lary=data_.FetchData(slice);
  switch (pooling_method_) {
    case EdgeProto::kMaxPooling:
      data_.set(-FLT_MAX);
      for (int n = nrng.first; n < nrng.second; n++) {
        for (int c = 0; c < channels_; c++) {
          for (int ph = 0; ph < pheight_; ph++) {
            for (int pw = 0; pw < pwidth_; pw++) {
              int hstart = ph * stride_;
              int wstart = pw * stride_;
              int hend = std::min(hstart + kernel_size_, height_);
              int wend = std::min(wstart + kernel_size_, width_);
              for (int h = hstart; h < hend; h++) {
                for (int w = wstart; w < wend; w++) {
                  data_.at(n,c,ph,pw)= std::max(data_.at(n,c,ph,pw),lary.get(n,c,h,w));
                }
              }
            }
          }
        }
      }
      break;
    case EdgeProto::kAvgPooling:
      data_.set(0.f);
      for (int n = rng.first; n < rng.second; n++) {
        for (int c = 0; c < channels_; c++) {
          for (int ph = 0; ph < pheight_; ph++) {
            for (int pw = 0; pw < pwidth_; pw++) {
              int hstart = ph * stride_;
              int wstart = pw * stride_;
              int hend = std::min(hstart + kernel_size_, height_);
              int wend = std::min(wstart + kernel_size_, width_);
              for (int h = hstart; h < hend; h++) {
                for (int w = wstart; w < wend; w++) {
                  data_.at(n,c,ph,pw) += lary.get(n,c,h,w);
                }
              }
              data_.at(n,c,ph,pw) /= (hend - hstart) * (wend - wstart);
            }
          }
        }
      }
      break;
    default:
      LOG(ERROR) << "Not supported pooling method ";
  }

}

/*
 * consider partition on num dim
 * assume grad and data have the same paritition
 */
void PoolingLayer::ComputeGradient() {
  const DAry& bottom=in_edges_[0]->GetData(this);
  DAry* gbottom=in_edges_[0]->GetMutableGrad(this);
  Range nrng=bottom_.IndexRange(0);
  vector<Range> slice{nrng, {0, channels_},{0, pheight_},{0,pwidth_}};
  DAry ldata=data_.FetchData(slice);
  DAry lgrad=grad_.FetchData(slice);
  switch (pooling_method_) {
    case EdgeProto::kMaxPooling:
      gbottom->set(0.f);
      for (int n = nrng.first; n < nrng.second; n++) {
        for (int c = 0; c < channels_; c++) {
          for (int ph = 0; ph < pheight_; ph++) {
            for (int pw = 0; pw < pwidth_; pw++) {
              int hstart = ph * stride_;
              int wstart = pw * stride_;
              int hend = std::min(hstart + kernel_size_, height_);
              int wend = std::min(wstart + kernel_size_, width_);
              for (int h = hstart; h < hend; h++) {
                for (int w = wstart; w < wend; w++) {
                  gbottom->at(n,c,h,w) += lgrad_.get(n,c,ph,pw)* (
                      bottom.get(n,c,h,w)==ldata.get(n,c,ph,pw));
                }
              }
            }
          }
        }
      }
      break;
    case EdgeProto::kAvgPooling:
      for (int n = nrng.first; n < nrng.second; n++) {
        for (int c = 0; c < channels_; c++) {
          for (int ph = 0; ph < pheight_; ph++) {
            for (int pw = 0; pw < pwidth_; pw++) {
              int hstart = ph * stride_;
              int wstart = pw * stride_;
              int hend = std::min(hstart + kernel_size_, height_);
              int wend = std::min(wstart + kernel_size_, width_);
              int count = (hend - hstart) * (wend - wstart);
              for (int h = hstart; h < hend; h++) {
                for (int w = wstart; w < wend; w++) {
                  gbottom->at(n,c,h,w) += lgrad.get(n,c,ph,pw) / count;
                }
              }
            }
          }
        }
      }
      break;
    default:
      LOG(ERROR) << "Not supported pooling method ";
  }

}


/*****************************************************************************
 * Implementation for LRNLayer
 *****************************************************************************/
void LRNLayer::Init(const LayerProto &proto) {
  Layer::Init(proto);
  wsize_ = proto.window_size();
  lpad_ = (wsize_ - 1) / 2;
  rpad_= wsize_-lpad_;
  alpha_ = proto.alpha();
  beta_ = proto.beta();
  knorm_=proto.knorm();
  if(proto.has_data())
    norm_.InitLike(data_);
}

void LRNLayer::ToProto(LayerProto* proto, bool copyData) {
  proto->set_window_size(wsize_);
  proto->set_alpha(alpha_);
  proto->set_beta(beta_);
  proto->set_knorm(knorm_);
}

void LRNLayer::SetupDAryShape(){
  const DAry& bottom=in_edges_[0]->GetData(this);
  CHECK_GE(bottom.Shape().size(),3);
  data_.SetShape(bottom.Shape());
  grad_.SetShape(bottom.Shape());
  norm_.SetShape(bottom.Shape());
}

void LRNLayer::AllocMemory(){
  data_.AllocMemory();
  grad_.AllocMemory();
  norm_.AllocMemory();
}
void LRNLayer::ComputeFeature() {
  VLOG(3)<<name_;
  Range nrng=data_.IdxRng(0);
  Range crng{0, data_.Shape(1)};
  std::vector<Range>slice{nrng, crng, {0, data_.Shape(2)}, {0, data_.Shape(3)}};
  const DAry& bottom=in_edges_[0]->GetData(this);
  const DAry& lbottom=bottom.FetchData(slice);
  // only share shape and partition not share data, allocate data here
  DAry squared3(lbottom[0], false);
  float alpha= alpha_ / window_;
  for(int n=nrng.first;n<nrng.second;++n) {
    DAry norm3=norm_[n];
    DAry bottom3=lbottom[n];
    DAry::Square(&squared3,bottom3);
    DAry::Mult(&squared3, squared3, alpha);
    // sum along 0-th dim, with range
    DAry::Sum(&norm3[0], squared3, 0, Range(0, rpad_));
    for(int c=1;c<channels;++c){
      DAry cur=norm3[c];
      DAry::Copy(cur, norm3[c-1]);
      if(c-lpad>=crng.first)
        DAry::Minus(&cur, cur, squared3[c-lpad]);
      if(c+rpad<=crng.second)
        DAry::Add(&cur, cur, squared3[c+rpad-1]);
    }
  }
  if(knorm_>0)
    DAry::Add(&norm_, norm_, knorm_);
  DAry::Pow(&data_, norm_, -beta_);
  DAry::Mult(&data_, data_, lbottom);
}

void LRNLayer::ComputeGradient() {
  VLOG(3)<<name_;
  float factor = -2.*alpha_ * beta_ / wsize_;
  DAry* gbottom=in_edges_[0]->GetMutableGrad(this);
  const DAry& bottom=in_edges_[0]->GetData(this);
  Range nrng=bottom.IdxRng(0);
  Range crng{0, bottom.Shape(1)};
  std::vector<Range>slice{nrng, crng, {0, bottom.Shape(2)},{0, bottom.Shape(3)}};
  DAry ldata=data_.FetchData(slice);
  DAry lgrad=grad_.FetchData(slice);
  DAry lnorm=norm_.FetchData(slice);
  DAry::Exp(gbottom, lnorm, -beta_);
  DAry::Mult(gbottom, *gbottom, lgrad);
  DAry ratio(bottom, false);
  DAry::Mult(&ratio, lgrad_, ldata_);
  DAry::Div(&ratio, ratio, lnorm);
  DAry accum_ratio(ratio[nrng.start][crng.first], false);
  for(int n=nrng.start;n<nrng.end;++n) {
    DAry gbottom3=(*gbottom)[n];
    DAry bottom3=bottom[n];
    DAry data3=ldata[n];
    DAry norm3=lnorm[n];
    DAry ratio3=lratio[n];
    DAry::Sum(&accum_ratio, lratio3,0, {0, lpad});
    for(int c=crng.first;c<crng.second;++c) {
      if(c+lpad<crng.second)
        DAry::Add(&accum2, accum2, ratio3[c+lpad]);
      DAry::Map(&gbottom3[c], [factor](float g, float a, float b)
          {return g-factor*a*b;}, gbottom3[c], accum_ratio, bottom3[c]);
      if(c-rpad+1>=0)
        DAry::Minus(&accum2, accum_ratio, ratio[c-rpad+1]);
    }
  }
}

/*****************************************************************************
 * Implementation for FCLayer
 *****************************************************************************/
void FCLayer::Init(const LayerProto &proto) {
  hdim_=proto.num_output();
  if(proto.has_weight())
    weight_.InitFromProto(proto.weight());
  if(proto.has_bias())
    bias_.InitFromProto(proto.bias());
}

void FCLayer::ToProto(LayerProto* proto, bool copyData) {
  ParamProto* weight=proto->set_weight();
  weight_.ToProto(weight, copyData);
  ParamProto* bias=proto->set_bias();
  bias_.ToProto(bias, copyData);
}
void Layer::CollectParams(vector<Param*> *params){
  weight_.set_id(params->size());
  params->push_back(&weight_);
  bias_.set_id(params->size());
  params->push_back(&bias_);
}

void FCLayer::SetupDAryShape(){
  const DAry& bottom=in_edges_[0]->GetData(this);
  num_=bottom.Shape(0);
  vdim_=bottom.Size()/num_;
  vector<int> shape{num_, hdim_};
  data_.SetShape(shape);
  grad_.SetShape(shape);
  weight_.SetShape(vdim_, hdim_);
  bias_.SetShape(hdim_);
}

void FCLayer::AllocMemory(){
  data_.AllocMemory();
  grad_.AllocMemory();
}
void FCLayer::ComputeFeature() {
  VLOG(3)<<name_;
  const DAry& bottom=in_edges_[0]->GetData(this);
  DAry data2(data_, {num, hdim_});
  DAry bottom2(bottom, {num, vdim_});
  DAry::Dot(&data2, bottom2, weight_.data());
}

void FCLayer::ComputeGradient() {
  VLOG(3)<<name_;
  const DAry& bottom=in_edges_[0]->GetData(this);
  DAry* gbottom=in_edges_[0]->GetMutableGrad(this);
  DAry bottom2(bottom, {num, vdim_});
  DAry grad2(grad_, {num, hdim_});

  DAry::Dot(weight_.mutable_grad(),bottom2, grad2, trans1=true, trans2=false);
  DAry::SumRows(bias_.mutable_grad(), grad2);

  // if dest_grad is nullptr, then we only compute gradients for parameters
  // this may happen when the lower layer is DataLayer
  if (gbottom != nullptr) {
    DAry gbottom2(*gbottom, {num, vdim_});
    DAry::Dot(gbottom2, grad2, weight_.data(), trans1=false, trans2=true);
  }
}


/*****************************************************************************
 * Implementation for SoftmaxLossLayer
 *****************************************************************************/
void SoftmaxLossLayer::Init(const LayerProto &proto) {
  dim_=proto.num_output();
  topk_=proto.topk();
}

void SoftmaxLossLayer::ToProto(LayerProto* proto, bool copyData) {
  Layer::Init(proto, copyData);
  proto->set_num_output(dim_);
  proto->set_topk(topk_);
}
void SoftmaxLossLayer::SetupDAryShape(){
  const DAry& bottom=in_edges_[0]->GetData(this);
  num_=bottom.Shape(0);
  CHECK_EQ(dim_,bottom.Size()/num_);
  vector<int> shape{num_, dim_};
  data_.SetShape(shape);
  grad_.SetShape(shape);
}

void SoftmaxLossLayer::AllocMemory(){
  data_.AllocMemory();
  grad_.AllocMemory();
}
void SoftmaxLossLayer::ComputeFeature() {
  VLOG(3)<<name_;
  Range nrng=data_.IndexRange(0);
  const DAry& bottom=in_edges_[0]->GetData(this);
  vector<Range> slice{nrng, {0,data_.SubShape().Size()}};
  DAry lbottom=bottom.FetchData(slice);
  for (int n = nrng.first; n < nrng.second; ++n) {
    DAry data=data_[n];
    float mmax = lbottom[n].max();
    DAry::Map(&data, [mmax](float v){return std::exp(v-mmax);}, lbottom[n]);
    float sum=data.Sum();
    DAry::Div(&data, data, sum);
  }
}

void SoftmaxLossLayer::ComputeGradient() {
  VLOG(3)<<name_;
  const DAry& label=in_edges_[1]->GetData(this);
  DAry* gbottom=in_edges_[0]->GetMutableGrad(this);
  DAry::Copy(gbottom, data_);
  Range nrng=gbottom->IndexRange(0);
  vector<Range> slice{nrng};
  DAry llabel=label.FetchData(slice);
  DAry gbottom2(*gbottom, {num, gbottom->Size()/num});
  for (int n = nrng.first; n < nrng.second; n++) {
    int k = static_cast<int>(llabel.get(n));
    // check gobottom2[n,k] on this node, 1 for 1-th dim
    if(gbottom2.Local(n,k))
      gbottom2.at(n, k) -= 1.f;
  }
  DAry::Div(gbottom, *gbottom, num_);
}

// assume only partition along 0-th dim, add perfs from all partition
Performance SoftmaxLossLayer::CalcPerf(bool loss, bool accuracy){
  int ncorrect=0;
  // DAry:SubShape returns the shape without the 0-th dimension
  int record_len=data_.Shape().SubShape().Size();
  VLOG(3)<<"calc perf, record len "<<record_len;
  float logprob=0.0f;
  const DAry& label=in_edges_[1]->GetData(this);
  Range nrng=data_.IndexRange(0);
  vector<Range> slice{nrng};
  DAry llabel=label.FetchData();
  for(int n=nrng.first;n<nrng.second;n++) {
    int label=static_cast<int>(llabel.get(n));
    CHECK(label>=0&&label<1000)<<"label "<<label;
    float prob_of_truth=data_.get(n,label);
    if(accuracy){
      int nlarger=0;
      // count num of probs larger than the prob of the ground truth label
      for(int i=0;i<record_len;i++) {
        if (data_.get(n,i)>prob_of_truth)
          nlarger++;
      }
      // if the ground truth is within the topk largest, this precdition is correct
      if(nlarger<=topk_)
        ncorrect++;
    }
    if(loss) {
      logprob-=log(std::max(prob_of_truth, kLogThreshold));
    }
  }
  VLOG(3)<<"end calc perf";
  Performance perf;
  perf.set_precision(ncorrect*1.0/num);
  perf.set_loss(logprob/num);
  return perf;
}

/****************************************************************************
 * Implementation for InputLayer
 ***************************************************************************/
void InputLayer::Init(const LayerProto& proto) {
  Layer::Init(proto);
  if(proto.has_data()){
    tmp_data_.InitFrom(proto.data());
  }
}
void InputLayer::SetInputData(DAry *data){
  if(data==nullptr)
    data_.SwapDptr(tmp_data_);
  else
    data_.SwapDptr(*data);
  offset_=0;
}

/*****************************************************************************
 * Implementation for ImageLayer
 *****************************************************************************/
void ImageLayer::Init(const LayerProto &proto) {
  InputLayer::Init(proto);
  cropsize_=proto.cropsize();
}

void ImageLayer::ToProto(LayerProto* proto, bool copyData) {
  InputLayer::ToProto(proto, copyData);
  proto->set_cropsize(cropsize_);
}
void ImageLayer::SetupDAryShape(const vector<vector<int>>& shapes){
  CHECK(shapes[0].size(),4);
  data_.SetShape(shapes[0]);
  tmp_data_.SetShape(shapes[0]);
}

void ImageLayer::AllocMemory(){
  data_.AllocMemory();
  tmp_data_.AllocMemory();
}

void ImageLayer::AddInputRecord(const Record &record){
  Range nrng=tmp_data_.IndexRange(0);
  CHECK_LT(offset_, nrng.second-nrng.first);
  int n=offset_+nrng.first;
  if (cropsize_) {
    float* data_dptr=data_.dptr;
    float* tmp_dptr=tmp_.dptr;
    // get a blob
    int h_off, w_off;
    // We only do random crop when we do training.
    if (Trainer::phase == Phase::kTrain) {
      h_off = rand() % (height_ - cropsize_);
      w_off = rand() % (width_ - cropsize_);
    } else {
      h_off = (height_ - cropsize_) / 2;
      w_off = (width_ - cropsize_) / 2;
    }
    // NOLINT_NEXT_LINE(runtime/threadsafe_fn)
    if (mirror_ && rand() % 2) {
      // Copy mirrored version
      for (int c = 0; c < channels_; ++c) {
        for (int h = 0; h < cropsize_; ++h) {
          for (int w = 0; w < cropsize_; ++w) {
            tmp_data_.at(n,c,h,cropsize_-1-w)=record.datum().value(
                (c * height_ + h + h_off) * width_ + w + w_off);
          }
        }
      }
    }
  } else {
    // Normal copy
    for (int c = 0; c < channels_; ++c) {
      for (int h = 0; h < cropsize_; ++h) {
        for (int w = 0; w < cropsize_; ++w) {
          tmp_data_.at(n,c,h,w)= record.datum().value(
              (c * height_+ h + h_off) * width_ + w + w_off);
        }
      }
    }
  }
  offset_++;
}

/*****************************************************************************
 * Implementation for LabelLayer
 *****************************************************************************/
void LabelLayer::SetupDAryShape(const vector<vector<int>>& shapes){
  CHECK(shapes.size(),2);
  CHECK(shapes[1].size(),2);
  data_.SetShape(shapes[1]);
  tmp_data_.SetShape(shapes[1]);
}

void LabelLayer::AllocMemory(){
  data_.AllocMemory();
  tmp_data_.AllocMemory();
}

void LabelLayer::AddInputRecord(const Record &record){
  Range nrng=tmp_data_.IndexRange(0);
  CHECK_LT(offset_, nrng.second-nrng.first);
  int n=offset_+nrng.first;
  tmp_data_.at(n)=static_cast<int>(record.label());
  offset_++;
}

/*****************************************************************************
 * Implementation for LayerFactory
 ****************************************************************************/
#define CreateLayer(LayerClass) [](void)->Layer* {return new LayerClass();}
std::shared_ptr<LayerFactory> LayerFactory::instance_;
std::shared_ptr<LayerFactory> LayerFactory::Instance() {
  if (!instance_.get()) {
    instance_.reset(new  LayerFactory());
  }
  return instance_;
}

LayerFactory::LayerFactory() {
  RegisterCreateFunction("ImageLayer", CreateLayer(ImageLayer));
  RegisterCreateFunction("LabelLayer", CreateLayer(LabelLayer));
  RegisterCreateFunction("ConvLayer", CreateLayer(ConvLayer));
  RegisterCreateFunction("ReLULayer", CreateLayer(ReLULayer));
  RegisterCreateFunction("PoolingLayer", CreateLayer(PoolingLayer));
  RegisterCreateFunction("LRNLayer", CreateLayer(LRNLayer));
  RegisterCreateFunction("FCLayer", CreateLayer(FCLayer));
  RegisterCreateFunction("SoftmaxLossLayer", CreateLayer(SoftmaxLossLayer));
}

void LayerFactory::RegisterCreateFunction(
  const std::string id,
  std::function<Layer*(void)> create_function) {
  layer_map_[id] = create_function;
}

Layer *LayerFactory::Create(const std::string id) {
  CHECK(layer_map_.find(id) != layer_map_.end())
      << "The initialization function " << id << " has not been registered";
  return layer_map_[id]();
}

}  // namespace lapis
