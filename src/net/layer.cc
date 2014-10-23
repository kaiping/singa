// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-06 15:19

#include <glog/logging.h>
#include <memory>
#include <cfloat>
#include "net/solver.h"
#include "net/layer.h"
namespace lapis {
/*****************************************************************************
 * Implementation for Layer
 *****************************************************************************/
void Layer::Init(const LayerProto &proto, StrStrEdge *edge_map) {
  name_ = proto.name();
  type_=proto.type();
  if(proto.has_data()){
    data_.InitFromProto(proto.data());
    grad_.InitFromProto(proto.data());
  }
  for(auto& top: proto.top()){
    auto p=std::make_pair(name_, top);
    Edge *e=nullptr;
    if(edge_map->find(p)!=edge_map->end()){
      e=edge_map->at(p);
      if(e->src()!=nullptr)
        CHECK(e->src()==this);
      else
        e->set_src(this);
    }else{
      e=new Edge();
      e->set_src(this);
      (*edge_map)[p]=e;
    }
    out_edges_.push_back(e);
  }
  for(auto& bottom: proto.bottom()){
    auto p=std::make_pair(bottom, name_);
    Edge* e=nullptr;
    if(edge_map->find(p)!=edge_map->end()){
      e=edge_map->at(p);
      if(e->dst()!=nullptr)
        CHECK(e->dst()==this);
      else
        e->set_dst(this);
    }else{
      e=new Edge();
      e->set_dst(this);
      (*edge_map)[p]=e;
    }
    in_edges_.push_back(e);
  }
}

void Layer::ToProto(LayerProto *proto, bool copyData) {
  proto->set_name(name_);
  proto->set_type(type_);
  DAryProto *data=proto->mutable_data();
  data_.ToProto(data, copyData);
  DAryProto *grad=proto->mutable_grad();
  grad_.ToProto(grad, copyData);
  proto->clear_bottom();
  for(auto* edge: in_edges_) {
    CHECK_EQ(edge->dst(), this);
    proto->add_bottom(edge->src()->name());
  }
  proto->clear_top();
  for(auto* edge: out_edges_) {
    CHECK_EQ(edge->src(), this);
    proto->add_top(edge->dst()->name());
  }
}

void Layer::InitDAryShape(const vector<vector<int>>& shapes){ }
void Layer::InitDAryShape(){}
void Layer::SetPartition(int pdim){
  data_.SetPartition(pdim);
  grad_.SetPartition(pdim);
}
void Layer::SetupDAry(int pdim){
  data_.Setup(pdim);
  grad_.Setup(pdim);
}
void Layer::ComputeFeature(){}
void Layer::ComputeGradient(){}
void Layer::CollectParams(vector<Param*> *params){}
vector<Param*> Layer::GetParams(){
  return vector<Param*>();
};
/*****************************************************************************
 * Implementation for ConvLayer
 *****************************************************************************/
void ConvLayer::Init(const LayerProto &proto,StrStrEdge *edge_map) {
  Layer::Init(proto, edge_map);
  CHECK(proto.has_window_size());
  wsize_ = proto.window_size();
  stride_ = proto.stride();
  pad_ = proto.pad();
  nkernels_ = proto.num_output();
  ngroups_ = proto.num_groups();
  if(proto.param().size()==2){
    weight_.Init(proto.param(0));
    bias_.Init(proto.param(1));
  }
  if(proto.has_col_data()){
    col_data_.InitFromProto(proto.col_data());
    col_grad_.InitFromProto(proto.col_grad());
  }
}
void ConvLayer::CollectParams(vector<Param*> *params){
  weight_.set_id(params->size());
  params->push_back(&weight_);
  bias_.set_id(params->size());
  params->push_back(&bias_);
}
vector<Param*> ConvLayer::GetParams() {
  vector<Param*> ret{&weight_, &bias_};
  return ret;
}
void ConvLayer::ToProto(LayerProto *proto, bool copyData) {
  Layer::ToProto(proto, copyData);
  proto->set_window_size(wsize_);
  proto->set_stride(stride_);
  proto->set_pad(pad_);
  proto->set_num_output(nkernels_);
  proto->set_num_groups(ngroups_);
  ParamProto* weight=proto->add_param();
  weight_.ToProto(weight, copyData);
  ParamProto* bias=proto->add_param();
  bias_.ToProto(bias, copyData);
  DAryProto* coldata=proto->mutable_col_data();
  DAryProto* colgrad=proto->mutable_col_grad();
  col_data_.ToProto(coldata, false);
  col_grad_.ToProto(colgrad, false);
}

void ConvLayer::InitDAryShape(){
  CHECK_EQ(in_edges_.size(), 1);
  CHECK_EQ(out_edges_.size(), 1);
  const DAry& bottom=in_edges_[0]->GetData(this);
  num_ = bottom.shape(0);
  channels_ = bottom.shape(1);
  height_ = bottom.shape(2);
  width_ = bottom.shape(3);
  // height and width of the image after convolution
  cheight_ = (height_ + 2 * pad_ - wsize_) / stride_ + 1;
  cwidth_ = (height_ + 2 * pad_ - wsize_) / stride_ + 1;
  vector<int> shape{num_, nkernels_, cheight_, cwidth_};
  data_.SetShape(shape);
  grad_.SetShape(shape);
  weight_.SetShape(nkernels_, wsize_*wsize_*channels_/ngroups_);
  bias_.SetShape(nkernels_);
  // weight matrix is of size nkernels_* K_, col_fea is of size
  // num_groups*K_*N_, image after conv is of shape (num_kernels_*N_)
  CHECK_EQ(nkernels_ % ngroups_ , 0)<< nkernels_<<", "<<ngroups_;
  CHECK_EQ((wsize_ * wsize_ * channels_) % ngroups_, 0)
    <<wsize_<<":"<<channels_<<":"<<ngroups_;
  M_ = nkernels_ / ngroups_;
  K_ = wsize_ * wsize_ * channels_ / ngroups_;
  N_ = cheight_ * cwidth_;
  col_data_.SetShape({num_, K_*ngroups_, cheight_, cwidth_});
  col_grad_.SetShape({num_, K_*ngroups_, cheight_, cwidth_});
  img2col=col2img=tdot=tadd=0;
}
void ConvLayer::SetPartition(int pdim){
  Layer::SetPartition(pdim);
  col_data_.SetPartition(pdim);
  col_grad_.SetPartition(pdim);
  weight_.SetPartition(-1);
  bias_.SetPartition(-1);
}
void ConvLayer::SetupDAry(int pdim){
  Layer::SetupDAry(pdim);
  weight_.SetupDAry(-1);
  bias_.SetupDAry(-1);
  col_data_.Setup(pdim);
  col_grad_.Setup(pdim);
}

void ConvLayer::ComputeFeature() {
  const DAry& bottom=in_edges_[0]->GetData(this);
  t.reset();
  Img2Col(&col_data_, bottom);
  img2col+=t.elapsed();
  // copy constructor with rvalue ref DAry(DAry&& other)
  // copy constructor with rvalue ref DAry(DAry&& other)
  DAry data4=data_.Reshape({num_, ngroups_, K_, N_});
  DAry col4=col_data_.Reshape({num_, ngroups_, M_, N_});
  DAry weight3=weight_.data().Reshape({ngroups_, M_, K_});
  Range nrng=data_.IndexRange(0);
  for (int n = nrng.first; n < nrng.second; n++) {
    // copy constructor with rvalue ref
    DAry data3=data4[n];
    DAry col3=col4[n];
    t.reset();
    for (int g = 0; g < ngroups_; g++){
      data3[g].Dot(weight3[g], col3[g]);
    }
    tdot+=t.elapsed();
    t.reset();
    DAry mat_data=data3.Reshape({nkernels_, N_});
    mat_data.AddCol(bias_.data());
    tadd+=t.elapsed();
  }
}

void ConvLayer::ComputeGradient() {
  {
    DAry *gbias=bias_.mutable_grad();
    DAry grad3=grad_.Reshape({num_, nkernels_, N_});
    // sum along 1-th dim, i.e., the result aray has length as the 1-th dim
    gbias->Fill(0.0f);
    for (int i = 0; i < num_; i++) {
      gbias->SumCol(grad3[i], false);
    }
  }
  const DAry weight3=weight_.data().Reshape({ngroups_, M_, K_});
  DAry gweight3=weight_.mutable_grad()->Reshape({ngroups_, M_, K_});
  DAry col4=col_data_.Reshape({num_, ngroups_,K_, N_});
  DAry* gbottom=in_edges_[0]->GetMutableGrad(this);
  DAry gcol4=col_grad_.Reshape({num_, ngroups_,K_, N_});
  if(gbottom!=nullptr){
    Range nrng=gbottom->IndexRange(0);
    // fetch grad_ and col
    for (int n = nrng.first; n < nrng.second; n++) {
      DAry grad3=grad_[n].Reshape({ngroups_,M_,N_});
      DAry col3=col4[n];
      DAry gcol3=gcol4[n];
      for (int g = 0; g < ngroups_; g++) {
        gweight3[g].Dot(grad3[g], col3[g], false, true);
        gcol3[g].Dot(weight3[g], grad3[g], true, false);
      }
    }
    Col2Img(gbottom, col_grad_);
  } else {
    t.reset();
    for (int n = 0; n < num_; n++) {
      DAry grad3=grad_[n].Reshape({ngroups_,M_,N_});
      DAry col3=col4[n];
      for (int g = 0; g < ngroups_; g++){
        gweight3[g].Dot(grad3[g], col3[g], false, true);
      }
    }
  }
}

/*
 * only consider parition along the num dimension
 */
void ConvLayer::Img2Col(DAry* dst, const DAry& src){
  const Range& nrng=dst->IndexRange(0);
  const Range& crng=dst->IndexRange(1);
  std::vector<Range> slice{nrng, crng,
    Range({0, height_}), Range({0, width_})};
  const DAry lsrc=src.Fetch(slice);
  //float* dstptr=dst->dptr();
  for(int n=nrng.first; n<nrng.second;n++){
    for (int c = crng.first; c < crng.second; ++c) {
      float* dptr=lsrc.addr(n,c,0,0);
      int w_offset = c % wsize_;
      int h_offset = (c / wsize_) % wsize_;
      int c_im = c / wsize_ / wsize_;
      for (int h = 0; h < cheight_; ++h) {
        for (int w = 0; w < cwidth_; ++w) {
          int h_pad = h * stride_ - pad_ + h_offset;
          int w_pad = w * stride_ - pad_ + w_offset;
          if (h_pad >= 0 && h_pad < height_ && w_pad >= 0 && w_pad < width_)
            *dptr=lsrc.at(n, c_im,h_pad, w_pad);
          else
            *dptr= 0;
          dptr++;
        }
      }
    }
  }
}
/*
 * consider only partition on num dimension
 */
void ConvLayer::Col2Img(DAry* dst, const DAry& src){
  Range nrng=dst->IndexRange(0);
  Range crng=dst->IndexRange(1);
  std::vector<Range> slice{nrng, crng,// {0, channels_*wsize_*wsize_},
    Range({0, cheight_}), Range({0, cwidth_})};
  DAry lsrc=src.Fetch(slice);
  lsrc.Fill(0.0f);
  // float *srcptr=lsrc.dptr();
  for(int n=nrng.first;n<nrng.second;n++){
    for (int c = crng.first; c < crng.second; ++c) {
      float* dptr=lsrc.addr(n,c,0,0);
      int w_offset = c % wsize_;
      int h_offset = (c / wsize_) % wsize_;
      int c_im = c / wsize_ / wsize_;
      for (int h = 0; h < cheight_; ++h) {
        for (int w = 0; w < cwidth_; ++w) {
          int h_pad = h * stride_ - pad_ + h_offset;
          int w_pad = w * stride_ - pad_ + w_offset;
          if (h_pad >= 0 && h_pad < height_ && w_pad >= 0 && w_pad < width_)
            dst->at(n, c_im,h_pad, w_pad) += *dptr;
          dptr++;
        }
      }
    }
  }
}

/*****************************************************************************
 * Implementation for ReLULayer
 *****************************************************************************/
void ReLULayer::InitDAryShape(){
  CHECK_EQ(in_edges_.size(),1);
  const DAry& bottom=in_edges_[0]->GetData(this);
  data_.SetShape(bottom.shape());
  grad_.SetShape(bottom.shape());
}

void ReLULayer::ComputeFeature() {
  data_.Max(in_edges_[0]->GetData(this), 0);
}

void ReLULayer::ComputeGradient() {
  DAry* gbottom=in_edges_[0]->GetMutableGrad(this);
  const DAry& bottom=in_edges_[0]->GetData(this);
  gbottom->Map([](float d, float g){return d>0?g:0;}, bottom, grad_);
}
/*****************************************************************************
 * Implementation for DropoutLayer
 *****************************************************************************/
void DropoutLayer::Init(const LayerProto &proto, StrStrEdge *edge_map) {
  Layer::Init(proto, edge_map);
  drop_prob_=proto.drop_prob();
  if(proto.has_data())
    mask_.InitFromProto(proto.data());
}

void DropoutLayer::ToProto(LayerProto *proto, bool copyData) {
  Layer::ToProto(proto, copyData);
  proto->set_drop_prob(drop_prob_);
}

void DropoutLayer::InitDAryShape(){
  CHECK_EQ(in_edges_.size(),1);
  const DAry& bottom=in_edges_[0]->GetData(this);
  data_.SetShape(bottom.shape());
  grad_.SetShape(bottom.shape());
  mask_.SetShape(bottom.shape());
}
void DropoutLayer::SetPartition(int pdim){
  Layer::SetPartition(pdim);
  mask_.SetPartition(pdim);
}

void DropoutLayer::SetupDAry(int pdim){
  Layer::SetupDAry(pdim);
  mask_.Setup(pdim);
}
void DropoutLayer::ComputeFeature() {
  float keep_prob = 1.0 - drop_prob_;
  mask_.Random();
  mask_.Threshold(mask_, keep_prob);
  //DAry::Map(&mask_, [keep_prob](float v){return v<=keep_prob?1.0f:0.0f;}, mask_);
  float scale=1.0/keep_prob;
  const DAry& bottom=in_edges_[0]->GetData(this);
  data_.Map([scale](float v, float m) {return v*m*scale;}, bottom, mask_);
}

void DropoutLayer::ComputeGradient() {
  DAry* gbottom=in_edges_[0]->GetMutableGrad(this);
  gbottom->Mult(grad_, mask_);
}
/*****************************************************************************
 * Implementation for PoolingLayer
 *****************************************************************************/
void PoolingLayer::Init(const LayerProto &proto, StrStrEdge *edge_map) {
  Layer::Init(proto, edge_map);
  wsize_ = proto.window_size();
  stride_ = proto.stride();
  pooling_method_ = proto.pooling_method();
}
void PoolingLayer::ToProto(LayerProto *proto, bool copyData) {
  Layer::ToProto(proto, copyData);
  proto->set_window_size(wsize_);
  proto->set_stride(stride_);
  proto->set_pooling_method(pooling_method_);
}
void PoolingLayer::InitDAryShape(){
  const DAry& bottom=in_edges_[0]->GetData(this);
  num_ = bottom.shape(0);
  channels_ = bottom.shape(1);
  height_ = bottom.shape(2);
  width_ = bottom.shape(3);
  pheight_ = static_cast<int> (
                   ceil(static_cast<float>(height_ - wsize_) / stride_)) + 1;
  pwidth_ = static_cast<int> (
                  ceil(static_cast<float>(width_ - wsize_) / stride_)) + 1;
  vector<int> shape{num_, channels_, pheight_, pwidth_};
  data_.SetShape(shape);
  grad_.SetShape(shape);
}
void PoolingLayer::ComputeFeature() {
  Range nrng=data_.IndexRange(0);
  Range crng=data_.IndexRange(1);
  vector<Range> slice{nrng, crng,Range({0, pheight_}),Range({0,pwidth_})};
  const DAry& bottom=in_edges_[0]->GetData(this);
  DAry lbottom=bottom.Fetch(slice);
  switch (pooling_method_) {
    case LayerProto::kMaxPooling:
      data_.Fill(-FLT_MAX);
      for (int n = nrng.first; n < nrng.second; n++) {
        for (int c = crng.first; c < crng.second; c++) {
          float* dptr=lbottom.addr(n,c,0,0);
          for (int ph = 0; ph < pheight_; ph++) {
            for (int pw = 0; pw < pwidth_; pw++) {
              int hstart = ph * stride_;
              int wstart = pw * stride_;
              int hend = std::min(hstart + wsize_, height_);
              int wend = std::min(wstart + wsize_, width_);
              for (int h = hstart; h < hend; h++) {
                for (int w = wstart; w < wend; w++) {
                  data_.at(n,c,ph,pw)= std::max(data_.at(n,c,ph,pw), *dptr);
                  dptr++;
                }
              }
            }
          }
        }
      }
      break;
    case LayerProto::kAvgPooling:
      data_.Fill(0.0f);
      for (int n = nrng.first; n < nrng.second; n++) {
        for (int c = crng.first; c < crng.second; c++) {
          float* dptr=lbottom.addr(n,c,0,0);
          for (int ph = 0; ph < pheight_; ph++) {
            for (int pw = 0; pw < pwidth_; pw++) {
              int hstart = ph * stride_;
              int wstart = pw * stride_;
              int hend = std::min(hstart + wsize_, height_);
              int wend = std::min(wstart + wsize_, width_);
              for (int h = hstart; h < hend; h++) {
                for (int w = wstart; w < wend; w++) {
                  data_.at(n,c,ph,pw) += *dptr;
                  dptr++;
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
  Range nrng=bottom.IndexRange(0);
  Range crng=bottom.IndexRange(1);
  vector<Range> slice{nrng, crng,Range({0, pheight_}),Range({0,pwidth_})};
  DAry ldata=data_.Fetch(slice);
  DAry lgrad=grad_.Fetch(slice);
  switch (pooling_method_) {
    case LayerProto::kMaxPooling:
      gbottom->Fill(0.0f);
      for (int n = nrng.first; n < nrng.second; n++) {
        for (int c = crng.first; c < crng.second; c++) {
          float* bgdptr=gbottom->addr(n,c,0,0);
          float* bdptr=bottom.addr(n,c,0,0);
          for (int ph = 0; ph < pheight_; ph++) {
            for (int pw = 0; pw < pwidth_; pw++) {
              int hstart = ph * stride_;
              int wstart = pw * stride_;
              int hend = std::min(hstart + wsize_, height_);
              int wend = std::min(wstart + wsize_, width_);
              for (int h = hstart; h < hend; h++) {
                for (int w = wstart; w < wend; w++) {
                  (*bgdptr) += lgrad.at(n,c,ph,pw)* (
                      *bdptr==ldata.at(n,c,ph,pw));
                  bgdptr++;
                  bdptr++;
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
void LRNLayer::Init(const LayerProto &proto, StrStrEdge *edge_map)  {
  Layer::Init(proto, edge_map);
  wsize_ = proto.window_size();
  lpad_ = (wsize_ - 1) / 2;
  rpad_= wsize_-lpad_;
  alpha_ = proto.alpha();
  beta_ = proto.beta();
  knorm_=proto.knorm();
  if(proto.has_data()){
    norm_.InitFromProto(proto.data());
    ratio_.InitFromProto(proto.data());
  }
}

void LRNLayer::ToProto(LayerProto* proto, bool copyData) {
  Layer::ToProto(proto, copyData);
  proto->set_window_size(wsize_);
  proto->set_alpha(alpha_);
  proto->set_beta(beta_);
  proto->set_knorm(knorm_);
}

void LRNLayer::InitDAryShape(){
  const DAry& bottom=in_edges_[0]->GetData(this);
  num_=bottom.shape(0);
  channels_=bottom.shape(1);
  height_=bottom.shape(2);
  width_=bottom.shape(3);
  CHECK_GE(bottom.shape().size,3);
  data_.SetShape(bottom.shape());
  grad_.SetShape(bottom.shape());
  norm_.SetShape(bottom.shape());
  ratio_.SetShape(bottom.shape());
}
void LRNLayer::SetPartition(int pdim){
  Layer::SetPartition(pdim);
  norm_.SetPartition(pdim);
  ratio_.SetPartition(pdim);
}

void LRNLayer::SetupDAry(int pdim){
  Layer::SetupDAry(pdim);
  norm_.Setup(pdim);
  ratio_.Setup(pdim);
}
void LRNLayer::ComputeFeature() {
  Range nrng=data_.IndexRange(0);
  Range crng({0, data_.shape(1)});
  std::vector<Range>slice{nrng, crng, Range({0, data_.shape(2)}), Range({0, data_.shape(3)})};
  const DAry& bottom=in_edges_[0]->GetData(this);
  const DAry& lbottom=bottom.Fetch(slice);
  // only share shape and partition not share data, allocate data here
  DAry squared3(lbottom.shape().SubShape());
  float alpha= alpha_ / wsize_;
  for(int n=nrng.first;n<nrng.second;++n) {
    DAry norm3=norm_[n];
    DAry bottom3=lbottom[n];
    squared3.Square(bottom3);
    squared3.Mult(squared3, alpha);
    norm3[0].Sum(squared3, Range(0, rpad_));
    for(int c=1;c<channels_;++c){
      DAry cur=norm3[c];
      cur.Copy(norm3[c-1]);
      if(c-lpad_>=0)
        cur.Minus(cur, squared3[c-lpad_]);
      if(c+rpad_<=crng.second)

        cur.Add(cur, squared3[c+rpad_-1]);
    }
  }
  if(knorm_>0)
    norm_.Add(norm_, knorm_);
  data_.Pow(norm_, -beta_);
  data_.Mult(data_, bottom);
}

void LRNLayer::ComputeGradient() {
  float factor = -2.*alpha_ * beta_ / wsize_;
  DAry* gbottom=in_edges_[0]->GetMutableGrad(this);
  const DAry& bottom=in_edges_[0]->GetData(this);
  Range nrng=bottom.IndexRange(0);
  Range crng({0, bottom.shape(1)});
  std::vector<Range>slice{nrng, crng, Range({0, bottom.shape(2)}),Range({0, bottom.shape(3)})};
  DAry ldata=data_.Fetch(slice);
  DAry lgrad=grad_.Fetch(slice);
  DAry lnorm=norm_.Fetch(slice);
  gbottom->Pow(lnorm, -beta_);
  gbottom->Mult(*gbottom, lgrad);
  ratio_.Mult(lgrad, ldata);
  ratio_.Div(ratio_, lnorm);
  DAry accum_ratio(ratio_.shape().SubShape().SubShape());
  for(int n=nrng.first;n<nrng.second;++n) {
    DAry gbottom3=(*gbottom)[n];
    DAry bottom3=bottom[n];
    DAry ratio3=ratio_[n];
    accum_ratio.Sum(ratio3, {0, lpad_});
    for(int c=crng.first;c<crng.second;++c) {
      if(c+lpad_<crng.second)
        accum_ratio.Add(ratio3[c+lpad_]);
      gbottom3[c].Map([factor](float g, float a, float b)
          {return g+factor*a*b;}, gbottom3[c], accum_ratio, bottom3[c]);
      if(c-rpad_+1>=0)
        accum_ratio.Minus(ratio3[c-rpad_+1]);
    }
  }
}

/*****************************************************************************
 * Implementation for FCLayer
 *****************************************************************************/
void FCLayer::Init(const LayerProto &proto,StrStrEdge *edge_map) {
  Layer::Init(proto, edge_map);
  hdim_=proto.num_output();
  CHECK(hdim_);
  if(proto.param().size()==2){
    weight_.Init(proto.param(0));
    bias_.Init(proto.param(1));
  }
}

void FCLayer::ToProto(LayerProto* proto, bool copyData) {
  Layer::ToProto(proto, copyData);
  ParamProto* weight=proto->add_param();
  weight_.ToProto(weight, copyData);
  ParamProto* bias=proto->add_param();
  bias_.ToProto(bias, copyData);
  proto->set_num_output(hdim_);
}
void FCLayer::CollectParams(vector<Param*> *params){
  weight_.set_id(params->size());
  params->push_back(&weight_);
  bias_.set_id(params->size());
  params->push_back(&bias_);
}
vector<Param*> FCLayer::GetParams() {
  vector<Param*> ret{&weight_, &bias_};
  return ret;
}

void FCLayer::InitDAryShape(){
  const DAry& bottom=in_edges_[0]->GetData(this);
  num_=bottom.shape(0);
  vdim_=bottom.shape().size/num_;
  vector<int> shape{num_, hdim_};
  data_.SetShape(shape);
  grad_.SetShape(shape);
  weight_.SetShape(vdim_, hdim_);
  bias_.SetShape(hdim_);
}
void FCLayer::SetupDAry(int pdim){
  Layer::SetupDAry(pdim);
  // partition parameters when doing model parallelism
  weight_.SetupDAry(1);
  bias_.SetupDAry(0);
}
void FCLayer::SetPartition(int pdim){
  Layer::SetPartition(pdim);
  weight_.SetPartition(1);
  bias_.SetPartition(0);
}

void FCLayer::ComputeFeature() {
  VLOG(3)<<name_;
  const DAry& bottom=in_edges_[0]->GetData(this);
  DAry bottom2=bottom.Reshape({num_, vdim_});
  data_.Dot(bottom2, weight_.data());
  data_.AddRow(bias_.data());
}

void FCLayer::ComputeGradient() {
  VLOG(3)<<name_;
  const DAry& bottom=in_edges_[0]->GetData(this);
  DAry* gbottom=in_edges_[0]->GetMutableGrad(this);
  DAry bottom2=bottom.Reshape({num_, vdim_});
  DAry grad2=grad_.Reshape({num_, hdim_});

  weight_.mutable_grad()->Dot(bottom2, grad2, true, false);
  bias_.mutable_grad()->SumRow(grad2, true);

  // if dest_grad is nullptr, then we only compute gradients for parameters
  // this may happen when the lower layer is DataLayer
  if (gbottom != nullptr) {
    DAry gbottom2=gbottom->Reshape({num_, vdim_});
    gbottom2.Dot(grad2, weight_.data(), false, true);
  }
}

Performance OutputLayer::CalcPerf(bool loss, bool accuracy){
  return Performance();
}
/*****************************************************************************
 * Implementation for SoftmaxLossLayer
 *****************************************************************************/
void SoftmaxLossLayer::Init(const LayerProto &proto,StrStrEdge *edge_map) {
  OutputLayer::Init(proto, edge_map);
  //dim_=proto.num_output();
  topk_=proto.topk();
}

void SoftmaxLossLayer::ToProto(LayerProto* proto, bool copyData) {
  Layer::ToProto(proto, copyData);
  proto->set_num_output(dim_);
  proto->set_topk(topk_);
}
void SoftmaxLossLayer::SetupDAry(int pdim) {
  Layer::SetupDAry(pdim);
}
void SoftmaxLossLayer::SetPartition(int pdim) {
  CHECK(pdim==-1);
  Layer::SetPartition(pdim);
}

void SoftmaxLossLayer::InitDAryShape(){
  const DAry& bottom=in_edges_[0]->GetData(this);
  num_=bottom.shape(0);
  CHECK_EQ(bottom.shape().size%num_,0);
  dim_=bottom.shape().size/num_;
  vector<int> shape{num_, dim_};
  data_.SetShape(shape);
  grad_.SetShape(shape);
}

void SoftmaxLossLayer::ComputeFeature() {
  /*
  if(!data_.allocated())
    return;
    */
  Range nrng=data_.IndexRange(0);
  const DAry& bottom=in_edges_[0]->GetData(this);
  vector<Range> slice{nrng, {0,data_.shape().SubShape().size}};
  DAry lbottom=bottom.Fetch(slice);
  for (int n = nrng.first; n < nrng.second; ++n) {
    float mmax = lbottom[n].Max();
    DAry data=data_[n];
    data.Map([mmax](float v){return std::exp(v-mmax);}, lbottom[n]);

    float sum=data.Sum();
    data.Div(data, sum);
  }
}

void SoftmaxLossLayer::ComputeGradient() {
  const DAry& label=in_edges_[1]->GetData(this);
  DAry* gbottom=in_edges_[0]->GetMutableGrad(this);
  gbottom->Copy(data_);
  Range nrng=gbottom->IndexRange(0);
  vector<Range> slice{nrng};
  DAry llabel=label.Fetch(slice);
  DAry gbottom2=gbottom->Reshape({num_, gbottom->shape().size/num_});
  for (int n = nrng.first; n < nrng.second; n++) {
    int k = static_cast<int>(llabel.at(n,0));
    // check gobottom2[n,k] on this node, 1 for 1-th dim
    if(gbottom2.isLocal(n,k))
      gbottom2.at(n,k)-=1.0f;
  }
  gbottom->Div(*gbottom, num_);
}

// assume only partition along 0-th dim, add perfs from all partition
Performance SoftmaxLossLayer::CalcPerf(bool loss, bool accuracy){
  int ncorrect=0;
  float logprob=0.0f;
  //if(data_.allocated()){
    // DAry:SubShape returns the shape without the 0-th dimension
    int record_len=data_.shape().SubShape().size;
    const DAry& label=in_edges_[1]->GetData(this);
    Range nrng=data_.IndexRange(0);
    vector<Range> slice{nrng};
    DAry llabel=label.Fetch(slice);
    for(int n=nrng.first;n<nrng.second;n++) {
      int label=static_cast<int>(llabel.at(n,0));
      CHECK(label>=0&&label<1000)<<"label "<<label;
      float prob_of_truth=data_.at(n,label);
      if(accuracy){
        int nlarger=0;
        // count num of probs larger than the prob of the ground truth label
        for(int i=0;i<record_len;i++) {
          if (data_.at(n,i)>prob_of_truth)
            nlarger++;
        }
        // if the ground truth is within the topk largest, this precdition is correct
        if(nlarger<=topk_)
          ncorrect++;
      }
      if(loss) {
        logprob-=log(std::max(prob_of_truth, FLT_MIN));
      }
    }
  //}
  Performance perf;
  perf.set_precision(ncorrect*1.0/num_);
  perf.set_loss(logprob/num_);
  return perf;
}

/****************************************************************************
 * Implementation for InputLayer
 ***************************************************************************/
void InputLayer::Init(const LayerProto& proto,StrStrEdge *edge_map){
  Layer::Init(proto, edge_map);
  offset_=0;
}
void InputLayer::SetInputData(DAry *data){
  if(data==nullptr)
    data_.SwapDptr(&grad_);
  else
    data_.SwapDptr(data);
  offset_=0;
}

/*****************************************************************************
 * Implementation for ImageLayer
 *****************************************************************************/
void ImageLayer::Init(const LayerProto &proto,StrStrEdge *edge_map) {
  InputLayer::Init(proto, edge_map);
  cropsize_=proto.cropsize();
  mirror_=proto.mirror();
}

void ImageLayer::ToProto(LayerProto* proto, bool copyData) {
  InputLayer::ToProto(proto, copyData);
  proto->set_cropsize(cropsize_);
  proto->set_mirror(mirror_);
}
void ImageLayer::InitDAryShape(const vector<vector<int>>& shapes){
  CHECK_EQ(shapes[0].size(),4);
  data_.SetShape(shapes[0]);
  grad_.SetShape(shapes[0]);
}
void ImageLayer::AddInputRecord(const Record &record){
  Range nrng=grad_.IndexRange(0);
  CHECK_LT(offset_, nrng.second-nrng.first);
  int n=offset_+nrng.first;
  const DAryProto& image=record.image();
  int channels=data_.shape(1);
  int height=data_.shape(2);
  int width=data_.shape(3);
  if (cropsize_) {
    // get a blob
    int h_off, w_off;
    // We only do random crop when we do training.
    if (Solver::phase == kTrain) {
      h_off = rand() % (height - cropsize_);
      w_off = rand() % (width - cropsize_);
    } else {
      h_off = (height - cropsize_) / 2;
      w_off = (width - cropsize_) / 2;
    }
    // NOLINT_NEXT_LINE(runtime/threadsafe_fn)
    if (mirror_ && rand() % 2) {
      // Copy mirrored version
      for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < cropsize_; ++h) {
          for (int w = 0; w < cropsize_; ++w) {
            grad_.at(n,c,h,cropsize_-1-w)=image.value(
                (c * height + h + h_off) * width + w + w_off);
          }
        }
      }
    }
    else {
      // Normal copy
      for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < cropsize_; ++h) {
          for (int w = 0; w < cropsize_; ++w) {
            grad_.at(n,c,h,w)= image.value(
                (c * height+ h + h_off) * width + w + w_off);
          }
        }
      }
    }
  }else{
    for (int c = 0; c < channels; ++c) {
      for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
          grad_.at(n,c,h,w)= image.value((c * height+ h ) * width + w);
        }
      }
    }
  }
  offset_++;
}

/*****************************************************************************
 * Implementation for LabelLayer
 *****************************************************************************/
void LabelLayer::InitDAryShape(const vector<vector<int>>& shapes){
  CHECK_EQ(shapes.size(),2);
  CHECK_EQ(shapes[1].size(),2);
  data_.SetShape(shapes[1]);
  grad_.SetShape(shapes[1]);
}

void LabelLayer::AddInputRecord(const Record &record){
  Range nrng=grad_.IndexRange(0);
  CHECK_LT(offset_, nrng.second-nrng.first);
  int n=offset_+nrng.first;
  grad_.at(n,0)=static_cast<int>(record.label());
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
  RegisterCreateFunction("DropoutLayer", CreateLayer(DropoutLayer));
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
