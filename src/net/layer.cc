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
  DArrayProto *data=proto->mutable_data();
  data_.ToProto(data, copyData);
  DArrayProto *grad=proto->mutable_grad();
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

void Layer::InitDArrayShape(const vector<vector<int>>& shapes){ }
void Layer::InitDArrayShape(){}
void Layer::AllocateMemory(){
  data_.Setup();
  grad_.Setup();
}
void Partition(int k=-1) {
  data_.set_mode(k);
  grad_.set_mode(k);
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
}

void ConvLayer::InitDArrayShape(){
  CHECK_EQ(in_edges_.size(), 1);
  CHECK_EQ(out_edges_.size(), 1);
  const DArray& bottom=in_edges_[0]->GetData(this);
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
  CHECK_EQ((wsize_ * wsize_ * channels_) % ngroups_, 0)<<wsize_<<":"<<channels_<<":"<<ngroups_;
  M_ = nkernels_ / ngroups_;
  K_ = wsize_ * wsize_ * channels_ / ngroups_;
  N_ = cheight_ * cwidth_;
  col_data_.SetShape({num_, K_*ngroups_, cheight_, cwidth_});
  col_grad_.SetShape({num_, K_*ngroups_, cheight_, cwidth_});
  img2col=col2img=tdot=tadd=0;
}

void Partition(int k=-1) {
  // partition on 0-th dim, i.e., records
  // partition on 1-th dim, i.e., channels
  CHECK(k<2);
  Layer::Partition(k);
  col_data_.set_mode(k);
  col_grad_.set_mode(k);
}
void ConvLayer::AllocateMemory(){
  data_.Setup();
  grad_.Setup();
  weight_.Setup();
  bias_.Setup();
  col_data_.Setup();
  col_grad_.Setup();
}

void ConvLayer::ComputeFeature() {
  const DArray& bottom=in_edges_[0]->GetData(this);
  t.reset();
  Img2Col(&col_data_, bottom);
  img2col+=t.elapsed();
  // copy constructor with rvalue ref DArray(DArray&& other)
  DArray data4=data_.Reshape(Shape({num_, ngroups_, M_, N_}));
  DArray bottom4=col_data_.Reshape(Shape({num_, ngroups_, K_, N_}));
  DArray weight3=weight_.data().Reshape(Shape({ngroups_, M_, K_}));
  for (int n = 0; n < num_; n++) {
    // copy constructor with rvalue ref
    DArray data3=data4[n];
    DArray bottom3=bottom4[n];
    t.reset();
    for (int g = 0; g < ngroups_; g++){
      data3[g].Dot(weight3[g], bottom3[g]);
    }
    tdot+=t.elapsed();
    t.reset();
    DArray mat_data(data3, {nkernels_, N_});
    mat_data.AddCol(bias_.data());
    tadd+=t.elapsed();
  }
}

void ConvLayer::ComputeGradient() {
  {
    t.reset();
    DArray *gbias=bias_.mutable_grad();
    DArray grad3=grad_.Reshape(Shape({num_, nkernels_, N_}));
    /*
    // sum along 1-th dim, i.e., the result aray has length as the 1-th dim
    gbias->Set(0.0f);
    for (int i = 0; i < num_; i++) {
      gbias->SumCol(grad3[i], false);
    }
    */
    gbias->SumExcept(grad3, 1);
    tadd+=t.elapsed();
  }

  const DArray weight3=weight_.data().Reshape(Shape({ngroups_, M_, K_}));
  DArray gweight3=weight_.mutable_grad()->Reshape(Shape({ngroups_, M_, K_}));
  DArray col4=col_data_.Reshape(Shape({num_, ngroups_,K_, N_}));
  DArray* gbottom=in_edges_[0]->GetMutableGrad(this);
  DArray gcol4=col_grad_.Reshape(Shape({num_, ngroups_,K_, N_}));
  if(gbottom!=nullptr){
    t.reset();
    for (int n = 0; n < num_; n++) {
      DArray grad3=grad_[n].Reshape(Shape({ngroups_,M_,N_}));
      DArray col3=col4[n];
      DArray gcol3=gcol4[n];
      for (int g = 0; g < ngroups_; g++) {
        gweight3[g].Dot(grad3[g], col3[g], false, true);
        gcol3[g].Dot(weight3[g], grad3[g], true, false);
      }
    }
    tdot+=t.elapsed();
    t.reset();
    Col2Img(gbottom, col_grad_);
    col2img+=t.elapsed();
  } else {
    t.reset();
    for (int n = 0; n < num_; n++) {
      DArray grad3=grad_[n].Reshape(Shape({ngroups_,M_,N_}));
      DArray col3=col4[n];
      for (int g = 0; g < ngroups_; g++){
        gweight3[g].Dot(grad3[g], col3[g], false, true);
      }
    }
    tdot+=t.elapsed();
  }
}

/*
 * only consider parition along the num dimension
 */
void ConvLayer::Img2Col(DArray* dst, const DArray& src){
  const Range& nrng=dst->IdxRng(0);
  const Range& crng=dst->IdxRng(1);
  std::vector<Range> slice{nrng, crng,
    Range({0, height_}), Range({0, width_})};
  const DArray lsrc=src.FetchData(slice);
  //float* dstptr=dst->dptr();
  for(int n=nrng.start(); n<nrng.end();n++){
    for (int c = crng.start(); c < crng.end(); ++c) {
      int w_offset = c % wsize_;
      int h_offset = (c / wsize_) % wsize_;
      int c_im = c / wsize_ / wsize_;
      for (int h = 0; h < cheight_; ++h) {
        for (int w = 0; w < cwidth_; ++w) {
          int h_pad = h * stride_ - pad_ + h_offset;
          int w_pad = w * stride_ - pad_ + w_offset;
          if (h_pad >= 0 && h_pad < height_ && w_pad >= 0 && w_pad < width_)
            //dst->at(n,c,h,w)
            //*dstptr=lsrc.get(n,c_im,h_pad, w_pad);
            dst->v(n,c,h,w)=lsrc.get(n,c_im,h_pad, w_pad);
          else
            //dst->at(n,c,h,w)= 0;
            //*dstptr=0;
            dst->v(n,c,h,w)=0;
          //dstptr++;
        }
      }
    }
  }
}
/*
 * consider only partition on num dimension
 */
void ConvLayer::Col2Img(DArray* dst, const DArray& src){
  Range nrng=dst->IdxRng(0);
  Range crng=dst->IdxRng(1);
  std::vector<Range> slice{nrng, crng,// {0, channels_*wsize_*wsize_},
    Range({0, cheight_}), Range({0, cwidth_})};
  DArray lsrc=src.FetchData(slice);
  lsrc.Set(0.0f);
  // float *srcptr=lsrc.dptr();
  for(int n=nrng.start();n<nrng.end();n++){
    for (int c = crng.start(); c < crng.end(); ++c) {
      int w_offset = c % wsize_;
      int h_offset = (c / wsize_) % wsize_;
      int c_im = c / wsize_ / wsize_;
      for (int h = 0; h < cheight_; ++h) {
        for (int w = 0; w < cwidth_; ++w) {
          int h_pad = h * stride_ - pad_ + h_offset;
          int w_pad = w * stride_ - pad_ + w_offset;
          if (h_pad >= 0 && h_pad < height_ && w_pad >= 0 && w_pad < width_)
            dst->v(n,c_im,h_pad, w_pad) += lsrc.v(n,c,h,w);//*srcptr;
          //srcptr++;
        }
      }
    }
  }
}

/*****************************************************************************
 * Implementation for ReLULayer
 *****************************************************************************/
void ReLULayer::InitDArrayShape(){
  CHECK_EQ(in_edges_.size(),1);
  const DArray& bottom=in_edges_[0]->GetData(this);
  data_.SetShape(bottom.shape());
  grad_.SetShape(bottom.shape());
}

void ReLULayer::ComputeFeature() {
  data_.Max(in_edges_[0]->GetData(this), 0);
}

void ReLULayer::ComputeGradient() {
  DArray* gbottom=in_edges_[0]->GetMutableGrad(this);
  const DArray& bottom=in_edges_[0]->GetData(this);
  gbottom->Map([](float d, float g){return d>0?g:0;}, bottom, grad_);
}
/*****************************************************************************
 * Implementation for DropoutLayer
 *****************************************************************************/
void DropoutLayer::Init(const LayerProto &proto, StrStrEdge *edge_map) {
  Layer::Init(proto, edge_map);
  drop_prob_=proto.drop_prob();
  if(proto.has_data());
    mask_.InitLike(data_);
}

void DropoutLayer::ToProto(LayerProto *proto, bool copyData) {
  Layer::ToProto(proto, copyData);
  proto->set_drop_prob(drop_prob_);
}

void DropoutLayer::InitDArrayShape(){
  CHECK_EQ(in_edges_.size(),1);
  const DArray& bottom=in_edges_[0]->GetData(this);
  data_.SetShape(bottom.shape());
  grad_.SetShape(bottom.shape());
  mask_.SetShape(bottom.shape());
}
void DropoutLayer::Partition(int k) {
  CHECK(k<2);
  Layer::Partition(k);
  mask_.set_mode(k);
}
void DropoutLayer::AllocateMemory(){
  data_.Setup();
  grad_.Setup();
  mask_.Setup();
}
void DropoutLayer::ComputeFeature() {
  float keep_prob = 1.0 - drop_prob_;
  mask_.Random();
  mask_.Threshold(mask_, keep_prob);
  //DArray::Map(&mask_, [keep_prob](float v){return v<=keep_prob?1.0f:0.0f;}, mask_);
  float scale=1.0/keep_prob;
  const DArray& bottom=in_edges_[0]->GetData(this);
  data_.Map([scale](float v, float m) {return v*m*scale;}, bottom, mask_);
}

void DropoutLayer::ComputeGradient() {
  DArray* gbottom=in_edges_[0]->GetMutableGrad(this);
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
void PoolingLayer::InitDArrayShape(){
  const DArray& bottom=in_edges_[0]->GetData(this);
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
  Range nrng=data_.IdxRng(0);
  Range crng=data_.IdxRng(1);
  vector<Range> slice{nrng, crng,Range({0, pheight_}),Range({0,pwidth_})};
  const DArray& bottom=in_edges_[0]->GetData(this);
  DArray lary=bottom.FetchData(slice);
  switch (pooling_method_) {
    case LayerProto::kMaxPooling:
      data_.Set(-FLT_MAX);
      for (int n = nrng.start(); n < nrng.end(); n++) {
        for (int c = crng.start(); c < crng.end(); c++) {
          for (int ph = 0; ph < pheight_; ph++) {
            for (int pw = 0; pw < pwidth_; pw++) {
              int hstart = ph * stride_;
              int wstart = pw * stride_;
              int hend = std::min(hstart + wsize_, height_);
              int wend = std::min(wstart + wsize_, width_);
              for (int h = hstart; h < hend; h++) {
                for (int w = wstart; w < wend; w++) {
                  //data_.at(n,c,ph,pw)= std::max(data_.at(n,c,ph,pw),lary.get(n,c,h,w));
                  data_.v(n,c,ph,pw)= std::max(data_.v(n,c,ph,pw),lary.v(n,c,h,w));
                }
              }
            }
          }
        }
      }
      break;
    case LayerProto::kAvgPooling:
      data_.Set(0.0f);
      for (int n = nrng.start(); n < nrng.end(); n++) {
        for (int c = crng.start(); c < crng.end(); c++) {
          for (int ph = 0; ph < pheight_; ph++) {
            for (int pw = 0; pw < pwidth_; pw++) {
              int hstart = ph * stride_;
              int wstart = pw * stride_;
              int hend = std::min(hstart + wsize_, height_);
              int wend = std::min(wstart + wsize_, width_);
              for (int h = hstart; h < hend; h++) {
                for (int w = wstart; w < wend; w++) {
                  //data_.at(n,c,ph,pw) += lary.get(n,c,h,w);
                  data_.v(n,c,ph,pw) += lary.v(n,c,h,w);
                }
              }
              //data_.at(n,c,ph,pw) /= (hend - hstart) * (wend - wstart);
              data_.v(n,c,ph,pw) /= (hend - hstart) * (wend - wstart);
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
  const DArray& bottom=in_edges_[0]->GetData(this);
  DArray* gbottom=in_edges_[0]->GetMutableGrad(this);
  Range nrng=bottom.IdxRng(0);
  Range crng=bottom.IdxRng(1);
  vector<Range> slice{nrng, crng,Range({0, pheight_}),Range({0,pwidth_})};
  DArray ldata=data_.FetchData(slice);
  DArray lgrad=grad_.FetchData(slice);
  switch (pooling_method_) {
    case LayerProto::kMaxPooling:
      gbottom->Set(0.0f);
      for (int n = nrng.start(); n < nrng.end(); n++) {
        for (int c = crng.start(); c < crng.end(); c++) {
          for (int ph = 0; ph < pheight_; ph++) {
            for (int pw = 0; pw < pwidth_; pw++) {
              int hstart = ph * stride_;
              int wstart = pw * stride_;
              int hend = std::min(hstart + wsize_, height_);
              int wend = std::min(wstart + wsize_, width_);
              for (int h = hstart; h < hend; h++) {
                for (int w = wstart; w < wend; w++) {
                  gbottom->v(n,c,h,w) += lgrad.v(n,c,ph,pw)* (
                      bottom.v(n,c,h,w)==ldata.v(n,c,ph,pw));
                }
              }
            }
          }
        }
      }
      break;
    case LayerProto::kAvgPooling:
      for (int n = nrng.start(); n < nrng.end(); n++) {
        for (int c = crng.start(); c < crng.end(); c++) {
          for (int ph = 0; ph < pheight_; ph++) {
            for (int pw = 0; pw < pwidth_; pw++) {
              int hstart = ph * stride_;
              int wstart = pw * stride_;
              int hend = std::min(hstart + wsize_, height_);
              int wend = std::min(wstart + wsize_, width_);
              int count = (hend - hstart) * (wend - wstart);
              for (int h = hstart; h < hend; h++) {
                for (int w = wstart; w < wend; w++) {
                  gbottom->v(n,c,h,w) += lgrad.v(n,c,ph,pw) / count;
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
  if(proto.has_data())
    norm_.InitLike(data_);
}

void LRNLayer::ToProto(LayerProto* proto, bool copyData) {
  Layer::ToProto(proto, copyData);
  proto->set_window_size(wsize_);
  proto->set_alpha(alpha_);
  proto->set_beta(beta_);
  proto->set_knorm(knorm_);
}

void LRNLayer::InitDArrayShape(){
  const DArray& bottom=in_edges_[0]->GetData(this);
  num_=bottom.shape(0);
  channels_=bottom.shape(1);
  height_=bottom.shape(2);
  width_=bottom.shape(3);
  CHECK_GE(bottom.shape().Size(),3);
  data_.SetShape(bottom.shape());
  grad_.SetShape(bottom.shape());
  norm_.SetShape(bottom.shape());
  ratio_.SetShape(bottom.shape());
}
void LRNLayer::Partition(int k) {
  CHECK(k<1);
  Layer::Partition(k);
  norm_.set_mode(k);
  ratio_.set_mode(k);
}
void LRNLayer::AllocateMemory(){
  Layer::AllocateMemory();
  norm_.Setup();
  ratio_.Setup();
}
void LRNLayer::ComputeFeature() {
  Range nrng=data_.IdxRng(0);
  Range crng({0, data_.shape(1)});
  std::vector<Range>slice{nrng, crng, Range({0, data_.shape(2)}), Range({0, data_.shape(3)})};
  const DArray& bottom=in_edges_[0]->GetData(this);
  const DArray& lbottom=bottom.FetchData(slice);
  // only share shape and partition not share data, allocate data here
  DArray squared3=DArray::Local(lbottom.shape().SubShape());
  float alpha= alpha_ / wsize_;
  for(int n=nrng.start();n<nrng.end();++n) {
    DArray norm3=norm_[n];
    DArray bottom3=lbottom[n];
    squared3.Square=DArray::Local(bottom3.shape());
    squared3.Mult(squared3, alpha);
    norm3[0].Sum(squared3, Range(0, rpad_));
    for(int c=1;c<channels_;++c){
      DArray cur=norm3[c];
      cur.Copy(norm3[c-1]);
      if(c-lpad_>=0)
        cur.Minus(cur, squared3[c-lpad_]);
      if(c+rpad_<=crng.end())
        cur.Add(cur, squared3[c+rpad_-1]);
    }
  }
  if(knorm_>0)
    norm_.Add( norm_, knorm_);
  data_.Pow(norm_, -beta_);
  data_.Mult(data_, lbottom);
}

void LRNLayer::ComputeGradient() {
  float factor = -2.*alpha_ * beta_ / wsize_;
  DArray* gbottom=in_edges_[0]->GetMutableGrad(this);
  const DArray& bottom=in_edges_[0]->GetData(this);
  Range nrng=bottom.IdxRng(0);
  Range crng({0, bottom.shape(1)});
  std::vector<Range>slice{nrng, crng, Range({0, bottom.shape(2)}),Range({0, bottom.shape(3)})};
  DArray ldata=data_.FetchData(slice);
  DArray lgrad=grad_.FetchData(slice);
  DArray lnorm=norm_.FetchData(slice);
  gbottom->Pow(lnorm, -beta_);
  gbottom->Mult(*gbottom, lgrad);
  ratio_.Mult(lgrad, ldata);
  ratio_.Div(ratio_, lnorm);
  DArray accum_ratio(ratio_.shape().SubShape().SubShape());
  for(int n=nrng.first;n<nrng.second;++n) {
    DArray gbottom3=(*gbottom)[n];
    DArray bottom3=bottom[n];
    DArray ratio3=ratio_[n];
    accum_ratio.Sum(ratio3, {0, lpad_});
    for(int c=crng.start();c<crng.end();++c) {
      if(c+lpad_<crng.end())
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

void FCLayer::InitDArrayShape(){
  const DArray& bottom=in_edges_[0]->GetData(this);
  num_=bottom.shape(0);
  vdim_=bottom.shape().Size()/num_;
  vector<int> shape{num_, hdim_};
  data_.SetShape(shape);
  grad_.SetShape(shape);
  weight_.SetShape(vdim_, hdim_);
  bias_.SetShape(hdim_);
}
void FCLayer::Partition(int k){
  Layer::Partition(k);
  if(k==1){
    weight_.Partition(1);
    bias_.Partition(0);
  }
}
void FCLayer::AllocateMemory(){
  weight_.Setup();
  bias_.Setup();
}
void FCLayer::ComputeFeature() {
  const DArray& bottom=in_edges_[0]->GetData(this);
  DArray data2=data_.ReShape(Shape({num_, hdim_}));
  DArray bottom2=bottom.Reshape(Shape({num_, vdim_}));
  data2.Dot(bottom2, weight_.data());
  data2.AddRow(bias_.data());
}

void FCLayer::ComputeGradient() {
  const DArray& bottom=in_edges_[0]->GetData(this);
  DArray* gbottom=in_edges_[0]->GetMutableGrad(this);
  DArray bottom2=bottom.Reshape(Shape({num_, vdim_}));
  DArray grad2=grad_.Reshape(Shape({num_, hdim_}));

  weight_.mutable_grad()->Dot(bottom2, grad2, true, false);
  bias_.mutable_grad()->SumRow(grad2);

  // if dest_grad is nullptr, then we only compute gradients for parameters
  // this may happen when the lower layer is DataLayer
  if (gbottom != nullptr) {
    DArray gbottom2=gbottom->Reshape(Shape({num_, vdim_}));
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
void SoftmaxLossLayer::Partition(int k) {
  CHECK(k==0);
  Layer::Partition(k);
}
void SoftmaxLossLayer::InitDArrayShape(){
  const DArray& bottom=in_edges_[0]->GetData(this);
  num_=bottom.shape(0);
  CHECK_EQ(bottom.shape().Size()%num_,0);
  dim_=bottom.shape().Size()/num_;
  vector<int> shape{num_, dim_};
  data_.SetShape(shape);
  grad_.SetShape(shape);
}

void SoftmaxLossLayer::ComputeFeature() {
  /*
  if(!data_.allocated())
    return;
    */
  Range nrng=data_.IdxRng(0);
  const DArray& bottom=in_edges_[0]->GetData(this);
  vector<Range> slice{nrng, {0,data_.shape().SubShape().Size()}};
  DArray lbottom=bottom.FetchData(slice);
  for (int n = nrng.first; n < nrng.second; ++n) {
    float mmax = lbottom[n].Max();
    DArray data=data_[n];
    data.Map([mmax](float v){return std::exp(v-mmax);}, lbottom[n]);
    float sum=data.Sum();
    data.Div(data, sum);
  }
}

void SoftmaxLossLayer::ComputeGradient() {
  const DArray& label=in_edges_[1]->GetData(this);
  DArray* gbottom=in_edges_[0]->GetMutableGrad(this);
  gbottom->Copy(data_);
  Range nrng=gbottom->IdxRng(0);
  vector<Range> slice{nrng};
  DArray llabel=label.FetchData(slice);
  DArray gbottom2(*gbottom, {num_, gbottom->shape().Size()/num_});
  for (int n = nrng.first; n < nrng.second; n++) {
    int k = static_cast<int>(llabel.get(n,0));
    // check gobottom2[n,k] on this node, 1 for 1-th dim
    //if(gbottom2.isLocal({n,k}))
    gbottom2.OpAt({n, k}, lambda x: x-1.f);
  }
  gbottom->Div(*gbottom, num_);
}

// assume only partition along 0-th dim, add perfs from all partition
Performance SoftmaxLossLayer::CalcPerf(bool loss, bool accuracy){
  int ncorrect=0;
  float logprob=0.0f;
  //if(data_.allocated()){
    // DArray:SubShape returns the shape without the 0-th dimension
    int record_len=data_.shape().SubShape().Size();
    const DArray& label=in_edges_[1]->GetData(this);
    Range nrng=data_.IdxRng(0);
    vector<Range> slice{nrng};
    DArray llabel=label.FetchData(slice);
    for(int n=nrng.first;n<nrng.second;n++) {
      int label=static_cast<int>(llabel.get(n,0));
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
  if(proto.has_data()){
    prefetch_data_.InitFromProto(proto.data());
  }
}
void InputLayer::SetInputData(DArray *data){
  if(data==nullptr)
    data_.SwapDptr(&prefetch_data_);
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
void ImageLayer::InitDArrayShape(const vector<vector<int>>& shapes){
  CHECK_EQ(shapes[0].size(),4);
  data_.SetShape(shapes[0]);
  prefetch_data_.SetShape(shapes[0]);
}
void ImageLayer::Partition(int k){
  data_.partition(k);
  prefetch_data_.partition(k);
}
void ImageLayer::AllocateMemory(){
  data_.Setup();
  prefetch_data_.Setup();
}
void ImageLayer::AddInputRecord(const Record &record){
  Range nrng=prefetch_data_.IdxRng(0);
  CHECK_LT(offset_, nrng.second-nrng.first);
  int n=offset_+nrng.first;
  const DArrayProto& image=record.image();
  if (cropsize_) {
    // get a blob
    int h_off, w_off;
    // We only do random crop when we do training.
    if (Solver::phase == kTrain) {
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
            prefetch_data_.v(n,c,h,cropsize_-1-w)=image.value(
                (c * height_ + h + h_off) * width_ + w + w_off);
          }
        }
      }
    }
    else {
      // Normal copy
      for (int c = 0; c < channels_; ++c) {
        for (int h = 0; h < cropsize_; ++h) {
          for (int w = 0; w < cropsize_; ++w) {
            prefetch_data_.v(n,c,h,w)= image.value(
                (c * height_+ h + h_off) * width_ + w + w_off);
          }
        }
      }
    }
  }else{
    for (int c = 0; c < channels_; ++c) {
      for (int h = 0; h < height_; ++h) {
        for (int w = 0; w < width_; ++w) {
          prefetch_data_.v(n,c,h,w)= image.value((c * height_+ h ) * width_ + w);
        }
      }
    }
  }
  offset_++;
}

/*****************************************************************************
 * Implementation for LabelLayer
 *****************************************************************************/
void LabelLayer::InitDArrayShape(const vector<vector<int>>& shapes){
  CHECK_EQ(shapes.size(),2);
  CHECK_EQ(shapes[1].size(),2);
  data_.SetShape(shapes[1]);
  prefetch_data_.SetShape(shapes[1]);
}
void LabelLayer::Partition(int k) {
  data_.partition(k);
  prefetch_data_.partition(k);
}
void LabelLayer::AllocateMemory(){
  data_.Setup();
  prefetch_data_.Setup();
}

void LabelLayer::AddInputRecord(const Record &record){
  Range nrng=prefetch_data_.IdxRng(0);
  CHECK_LT(offset_, nrng.second-nrng.first);
  int n=offset_+nrng.first;
  prefetch_data_.at(n,0)=static_cast<int>(record.label());
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
