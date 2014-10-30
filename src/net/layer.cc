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
bool Layer::PreSyncF(){
  if(in_edges_.size()==0)
    return false;
  const DAry& bottom=in_edges_[0]->GetData(this);
  if(bottom.GetPartition()==-1||bottom.GetPartition()==data_.GetPartition())
    return false;
  else
    return true;
}
bool Layer::PreSyncG(){
  if(in_edges_.size()==0)
    return false;
  const DAry& bottom=in_edges_[0]->GetData(this);
  if(data_.GetPartition()==-1||bottom.GetPartition()==data_.GetPartition())
    return false;
  else
    return true;
}

void Layer::CollectParams(vector<Param*> *params){}
vector<Param*> Layer::GetParams(){
  return vector<Param*>();
};
/*****************************************************************************
 * Implementation for SplitLayer
 *****************************************************************************/
void SplitLayer::Init(const LayerProto &proto,StrStrEdge *edge_map) {
  Layer::Init(proto, edge_map);
  CHECK(in_edges_.size());
  split_dim_=proto.split_dim();
  split_size_=proto.split_size();
  if(proto.has_split_data()){
    data2_.InitFromProto(proto.split_data());
    grad2_.InitFromProto(proto.split_data());
  }
}

void SplitLayer::SetPartition(int pdim){
  Layer::SetPartition(pdim);
  data2_.SetPartition(pdim);
  grad2_.SetPartition(pdim);
}

void SplitLayer::SetupDAry(int pdim){
  Layer::SetupDAry(pdim);
  data2_.Setup(pdim);
  grad2_.Setup(pdim);
}
void SplitLayer::ToProto(LayerProto *proto, bool copyData) {
  Layer::ToProto(proto, copyData);
  DAryProto* dproto=proto->mutable_split_data();
  data2_.ToProto(dproto, false);
  proto->set_split_dim(split_dim_);
  proto->set_split_size(split_size_);
}
void SplitLayer::InitDAryShape(){
  CHECK_EQ(in_edges_.size(), 1);
  CHECK_EQ(out_edges_.size(), 2);
  const DAry& bottom=in_edges_[0]->GetData(this);
  int num, height, width, first_split=split_size_, second_split;
  if(split_dim_==0){
    LOG(ERROR)<<"Not implemented";
  } else{
    num=bottom.shape(0);
    second_split=bottom.shape(1)-first_split;
  }
  height = bottom.shape(2);
  width = bottom.shape(3);

  vector<int> shape1{num, first_split, height, width};
  data_.SetShape(shape1);
  grad_.SetShape(shape1);
  vector<int> shape2{num, second_split, height, width};
  data2_.SetShape(shape2);
  grad2_.SetShape(shape2);
}
bool SplitLayer::PreSyncF(){
  const DAry& bottom=in_edges_[0]->GetData(this);
  CHECK_NE(bottom.GetPartition()+data_.GetPartition(),1)<<
    "Not supported for diff parallelism modes";
  // bottom is replicated/local, or both are data parallel
  if(bottom.GetPartition()==-1||
      (bottom.GetPartition()==0&&data_.GetPartition()==0))
    return false;
  else
    return true;
}
bool SplitLayer::PreSyncG(){
  if(in_edges_.size()==0)
    return false;
  const DAry& bottom=in_edges_[0]->GetData(this);
  CHECK_NE(bottom.GetPartition()+data_.GetPartition(),1)<<
    "Not supported for diff parallelism modes";
  // bottom is replicated/local, or both are data parallel
  if(data_.GetPartition()==-1||
      (bottom.GetPartition()==0&&data_.GetPartition()==0))
    return false;
  else
    return true;
}

void SplitLayer::ComputeFeature() {
  const DAry& bottom=in_edges_[0]->GetData(this);
  if(split_dim_==0){
    LOG(ERROR)<<"Not implemented";
  }else{
    const Shape& s0=data_.shape();
    const Shape& s1=data2_.shape();
    int a=s0.size/s0.s[0];
    int b=a+s1.size/s1.s[0];
    data_.CopyFromCols(0, a, bottom);
    data2_.CopyFromCols(a, b, bottom);
  }
}

void SplitLayer::ComputeGradient() {
  DAry* gbottom=in_edges_[0]->GetMutableGrad(this);
  if(split_dim_==0){
    //data_.CopyToRows(0,top0.shape(0),top0);
    //data_.CopyToRows(top0.shape(0),num_, top1);
    LOG(ERROR)<<"Not implemented";
  }else{
    const Shape& s0=data_.shape();
    const Shape& s1=data2_.shape();
    int a=s0.size/s0.s[0];
    int b=a+s1.size/s1.s[0];
    gbottom->CopyToCols(0, a, grad_);
    gbottom->CopyToCols(a, b, grad2_);
  }
}
/*****************************************************************************
 * Implementation for ConcatLayer
 *****************************************************************************/
void ConcatLayer::Init(const LayerProto &proto,StrStrEdge *edge_map) {
  Layer::Init(proto, edge_map);
  concat_dim_=proto.concat_dim();
}
void ConcatLayer::ToProto(LayerProto *proto, bool copyData) {
  Layer::ToProto(proto, copyData);
  proto->set_concat_dim(concat_dim_);
}
void ConcatLayer::InitDAryShape(){
  CHECK_EQ(in_edges_.size(), 2);
  CHECK_EQ(out_edges_.size(), 1);
  const DAry& bottom0=in_edges_[0]->GetData(this);
  const DAry& bottom1=in_edges_[1]->GetData(this);
  int num, channels, height, width;
  if(concat_dim_==0){
    num = bottom0.shape(0)+bottom1.shape(0);;
    channels = bottom0.shape(1);
    CHECK_EQ(channels, bottom1.shape(1));
  } else{
    num=bottom0.shape(0);
    CHECK_EQ(num, bottom1.shape(0));
    channels=bottom0.shape(1)+bottom1.shape(1);
  }
  height = bottom0.shape(2);
  width = bottom0.shape(3);
  CHECK_EQ(height, bottom1.shape(2));
  CHECK_EQ(width, bottom1.shape(3));

  vector<int> shape{num, channels, height, width};
  data_.SetShape(shape);
  grad_.SetShape(shape);
}
bool ConcatLayer::PreSyncF(){
  const DAry& bottom=in_edges_[0]->GetData(this);
  CHECK_NE(bottom.GetPartition()+data_.GetPartition(),1)<<
    "Not supported for diff parallelism modes";
  // bottom is replicated/local, or both are data parallel
  if(bottom.GetPartition()==-1||
      (bottom.GetPartition()==0&&data_.GetPartition()==0))
    return false;
  else
    return true;
}
bool ConcatLayer::PreSyncG(){
  if(in_edges_.size()==0)
    return false;
  const DAry& bottom=in_edges_[0]->GetData(this);
  CHECK_NE(bottom.GetPartition()+data_.GetPartition(),1)<<
    "Not supported for diff parallelism modes";
  // bottom is replicated/local, or both are data parallel
  if(data_.GetPartition()==-1||
      (bottom.GetPartition()==0&&data_.GetPartition()==0))
    return false;
  else
    return true;
}
void ConcatLayer::ComputeFeature() {
  const DAry& bottom0=in_edges_[0]->GetData(this);
  const DAry& bottom1=in_edges_[1]->GetData(this);

  if(concat_dim_==0){
    //data_.CopyToRows(0,bottom0.shape(0),bottom0);
    //data_.CopyToRows(bottom0.shape(0),num_, bottom1);
    LOG(ERROR)<<"Not implemented";
  }else{
    const Shape& s0=bottom0.shape();
    const Shape& s1=bottom1.shape();
    int a=s0.size/s0.s[0];
    int b=a+s1.size/s1.s[0];
    data_.CopyToCols(0, a, bottom0);
    data_.CopyToCols(a, b, bottom1);
  }
}

void ConcatLayer::ComputeGradient() {
  DAry* gbottom0=in_edges_[0]->GetMutableGrad(this);
  DAry* gbottom1=in_edges_[1]->GetMutableGrad(this);

  if(concat_dim_==0){
    //data_.CopyToRows(0,bottom0.shape(0),bottom0);
    //data_.CopyToRows(bottom0.shape(0),num_, bottom1);
    LOG(ERROR)<<"Not implemented";
  }else{
    const Shape& s0=gbottom0->shape();
    const Shape& s1=gbottom1->shape();
    int a=s0.size/s0.s[0];
    int b=a+s1.size/s1.s[0];
    gbottom0->CopyFromCols(0, a, grad_);
    gbottom1->CopyFromCols(a, b, grad_);
  }
}

/*****************************************************************************
 * Implementation for ImgColLayer
 *****************************************************************************/
void ImgColLayer::Init(const LayerProto &proto,StrStrEdge *edge_map) {
  Layer::Init(proto, edge_map);
  CHECK_EQ(in_edges_.size(), 1);
  CHECK_EQ(out_edges_.size(), 1);
  CHECK(proto.has_window_size());
  wsize_ = proto.window_size();
  stride_ = proto.stride();
  pad_ = proto.pad();
}
void ImgColLayer::ToProto(LayerProto *proto, bool copyData) {
  Layer::ToProto(proto, copyData);
  proto->set_window_size(wsize_);
  proto->set_stride(stride_);
  proto->set_pad(pad_);
}

void ImgColLayer::InitDAryShape(){
  const DAry& bottom=in_edges_[0]->GetData(this);
  num_ = bottom.shape(0);
  channels_ = bottom.shape(1);
  height_ = bottom.shape(2);
  width_ = bottom.shape(3);
  // height and width of the image after convolution
  cheight_ = (height_ + 2 * pad_ - wsize_) / stride_ + 1;
  cwidth_ = (height_ + 2 * pad_ - wsize_) / stride_ + 1;
  vector<int> shape{num_, channels_*wsize_*wsize_, cheight_, cwidth_};
  data_.SetShape(shape);
  grad_.SetShape(shape);
}

/*
 * only consider parition along the num/channel dimension
 */
void ImgColLayer::Img2Col(DAry* dst, const DAry& src){
  CHECK_EQ(src.shape().dim,4);
  CHECK_EQ(dst->shape().dim,4);

  const Range& nrng=dst->IndexRange(0);
  const Range& dstcrng=dst->IndexRange(1);
  int w2=wsize_*wsize_;
  CHECK_EQ(dstcrng.first%w2,0);
  CHECK_EQ(dstcrng.second%w2,0);
  Range srccrng(dstcrng.first/w2, dstcrng.second/w2);
  vector<Range> slice{nrng, srccrng, Range({0, height_}), Range({0, width_})};
  const DAry& lsrc=src.Fetch(slice);
  //float* dstptr=dst->dptr();
  for(int n=nrng.first; n<nrng.second;++n){
    for (int c = dstcrng.first; c < dstcrng.second; ++c) {
      int w_offset = c % wsize_;
      int h_offset = (c / wsize_) % wsize_;
      int c_im = c / wsize_ / wsize_;
      float* dptr=dst->addr(n,c,0,0);
      float* sdptr=lsrc.addr(n,c_im,0,0);
      for (int h = 0; h < cheight_; ++h) {
        for (int w = 0; w < cwidth_; ++w) {
          int h_pad = h * stride_ - pad_ + h_offset;
          int w_pad = w * stride_ - pad_ + w_offset;
          if (h_pad >= 0 && h_pad < height_ && w_pad >= 0 && w_pad < width_)
            *dptr=sdptr[h_pad*width_+w_pad];//lsrc.at(n, c_im,h_pad, w_pad);
          else
            *dptr= 0;
          dptr++;
        }
      }
    }
  }
}

/*
 * consider only partition on num/channels dimension
 */
void ImgColLayer::Col2Img(DAry* dst, const DAry& src){
  Range nrng=dst->IndexRange(0);
  Range crng=dst->IndexRange(1);
  Range srccrng(crng.first*wsize_*wsize_, crng.second*wsize_*wsize_);
  vector<Range> slice{nrng, srccrng, Range({0, cheight_}), Range({0, cwidth_})};
  const DAry& lsrc=src.Fetch(slice);
  dst->Fill(0.0f);
  // float *srcptr=lsrc.dptr();
  for(int n=nrng.first;n<nrng.second;n++){
    for (int c = srccrng.first; c < srccrng.second; ++c) {
      int w_offset = c % wsize_;
      int h_offset = (c / wsize_) % wsize_;
      int c_im = c / wsize_ / wsize_;
      float *ddptr=dst->addr(n, c_im, 0,0);
      float* dptr=lsrc.addr(n,c,0,0);
      for (int h = 0; h < cheight_; ++h) {
        for (int w = 0; w < cwidth_; ++w) {
          int h_pad = h * stride_ - pad_ + h_offset;
          int w_pad = w * stride_ - pad_ + w_offset;
          if (h_pad >= 0 && h_pad < height_ && w_pad >= 0 && w_pad < width_)
            ddptr[h_pad*width_+w_pad]+=*dptr;
            //dst->at(n, c_im,h_pad, w_pad) += *dptr;
          dptr++;
        }
      }
    }
  }
}

void ImgColLayer::ComputeFeature() {
  const DAry& bottom=in_edges_[0]->GetData(this);
  Img2Col(&data_, bottom);
}

void ImgColLayer::ComputeGradient() {
  DAry* gbottom=in_edges_[0]->GetMutableGrad(this);
  if(gbottom!=nullptr)
    Col2Img(gbottom, grad_);
}

/*****************************************************************************
 * Implementation for ConvProductLayer
 *****************************************************************************/
void ConvProductLayer::Init(const LayerProto &proto, StrStrEdge *edge_map) {
  Layer::Init(proto, edge_map);
  kernels_=proto.num_output();
  CHECK_GT(kernels_,0);
  CHECK_EQ(proto.param().size(),2);
  weight_.Init(proto.param(0));
  bias_.Init(proto.param(1));
}

void ConvProductLayer::ToProto(LayerProto* proto, bool copyData) {
  Layer::ToProto(proto, copyData);
  ParamProto* weight=proto->add_param();
  weight_.ToProto(weight, copyData);
  ParamProto* bias=proto->add_param();
  bias_.ToProto(bias, copyData);
  proto->set_num_output(kernels_);
}
void ConvProductLayer::CollectParams(vector<Param*> *params){
  weight_.set_id(params->size());
  params->push_back(&weight_);
  bias_.set_id(params->size());
  params->push_back(&bias_);
}
vector<Param*> ConvProductLayer::GetParams() {
  vector<Param*> ret{&weight_, &bias_};
  return ret;
}

void ConvProductLayer::InitDAryShape(){
  const DAry& bottom=in_edges_[0]->GetData(this);
  num_=bottom.shape(0);
  channels_=bottom.shape(1);
  height_=bottom.shape(2);
  width_=bottom.shape(3);
  vector<int> shape{num_, kernels_, height_, width_};
  data_.SetShape(shape);
  grad_.SetShape(shape);
  weight_.SetShape(kernels_, channels_);
  bias_.SetShape(kernels_);
}
void ConvProductLayer::SetupDAry(int pdim){
  Layer::SetupDAry(pdim);
  // partition parameters when doing model parallelism
  if(pdim==0||pdim==-1){
    weight_.SetupDAry(-1);
    bias_.SetupDAry(-1);
  }else if(pdim==1){
    weight_.SetupDAry(0);
    bias_.SetupDAry(0);
  }else{
    LOG(ERROR)<<"Not supported partition dim "<<pdim;
  }
}
void ConvProductLayer::SetPartition(int pdim){
  Layer::SetPartition(pdim);
  if(pdim==-1||pdim==0){
    weight_.SetPartition(-1);
    bias_.SetPartition(-1);
  }else if(pdim==1){
    weight_.SetPartition(0);
    bias_.SetPartition(0);
  }else{
    LOG(ERROR)<<"Not supported partition dim "<<pdim;
  }
}
bool ConvProductLayer::PreSyncF(){
  const DAry& bottom=in_edges_[0]->GetData(this);
  if((bottom.GetPartition()==0&&data_.GetPartition()==0)
      ||bottom.GetPartition()==-1)
    return false;
  else
    return true;
}
bool ConvProductLayer::PreSyncG(){
  const DAry& bottom=in_edges_[0]->GetData(this);
  if((bottom.GetPartition()==0&&data_.GetPartition()==0)
      ||data_.GetPartition()==-1)
    return false;
  else
    return true;
}
bool ConvProductLayer::PostSyncF(){
  return PreSyncG();
}
bool ConvProductLayer::PostSyncG(){
  return PreSyncF();
}

void ConvProductLayer::ComputeFeature() {
  const Range nrng=data_.IndexRange(0);
  const DAry& bottom=in_edges_[0]->GetData(this);
  //vector<Range> slice{nrng, }
  DAry bottom3=bottom.Reshape({num_, channels_, height_*width_});
  DAry data=data_.Reshape({num_, kernels_, height_*width_});
  for(int n=nrng.first;n<nrng.second;n++){
    DAry image=data[n];
    image.Dot(weight_.data(), bottom3[n]);
    image.AddCol(bias_.data());
  }
}

void ConvProductLayer::ComputeGradient() {
  DAry* gbottom=in_edges_[0]->GetMutableGrad(this);
  DAry gbottom3=gbottom->Reshape({num_, channels_, height_*width_});
  const DAry& bottom=in_edges_[0]->GetData(this);
  DAry bottom3=bottom.Reshape({num_, channels_, height_*width_});
  DAry data=data_.Reshape({num_, kernels_, height_*width_});
  DAry grad=grad_.Reshape({num_, kernels_, height_*width_});
  const Range nrng=gbottom->IndexRange(0);
  DAry* gweight=weight_.mutable_grad();
  gweight->Fill(0.f);
  DAry* gbias=bias_.mutable_grad();
  gbias->Fill(0.f);
  for(int n=nrng.first;n<nrng.second;n++){
    DAry image_grad=gbottom3[n];
    DAry gradn=grad[n];
    image_grad.Dot(weight_.data(), gradn, true, false, true);
    gweight->Dot(gradn, bottom3[n], false, true, false);
    gbias->SumCol(gradn, false);
  }
}


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
    col_grad_.InitFromProto(proto.col_data());
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
  col_data_.ToProto(coldata, false);
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
  if(pdim==1)
    pdim=-1;
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
  Img2Col(&col_data_, bottom);

  DAry data4=data_.Reshape({num_, ngroups_, M_, N_});
  DAry col4=col_data_.Reshape({num_, ngroups_, K_, N_});
  DAry weight3=weight_.data().Reshape({ngroups_, M_, K_});
  Range nrng=data_.IndexRange(0);
  for (int n = nrng.first; n < nrng.second; n++) {
    // copy constructor with rvalue ref
    DAry data3=data4[n];
    DAry col3=col4[n];
    //t.reset();
    for (int g = 0; g < ngroups_; g++){
      data3[g].Dot(weight3[g], col3[g]);
    }
    //t.reset();
    DAry mat_data=data3.Reshape({nkernels_, N_});
    mat_data.AddCol(bias_.data());
  }
}

void ConvLayer::ComputeGradient() {
  {
    DAry *gbias=bias_.mutable_grad();
    DAry grad3=grad_.Reshape({num_, nkernels_, N_});
    // sum along 1-th dim, i.e., the result aray has length as the 1-th dim
    gbias->Fill(0.0f);
    Range rng=grad_.IndexRange(0);
    for (int i = rng.first; i < rng.second; i++) {
      gbias->SumCol(grad3[i], false);
    }
  }
  const DAry weight3=weight_.data().Reshape({ngroups_, M_, K_});
  DAry gweight3=weight_.mutable_grad()->Reshape({ngroups_, M_, K_});
  DAry col4=col_data_.Reshape({num_, ngroups_,K_, N_});
  DAry* gbottom=in_edges_[0]->GetMutableGrad(this);
  DAry gcol4=col_grad_.Reshape({num_, ngroups_,K_, N_});
  DAry grad4=grad_.Reshape({num_, ngroups_,M_,N_});
  gweight3.Fill(0.f);
  if(gbottom!=nullptr){
    Range nrng=gbottom->IndexRange(0);
    // fetch grad_ and col
    for (int n = nrng.first; n < nrng.second; n++) {
      DAry grad3=grad4[n];
      DAry col3=col4[n];
      DAry gcol3=gcol4[n];
      for (int g = 0; g < ngroups_; g++) {
        gweight3[g].Dot(grad3[g], col3[g], false, true, false);
        gcol3[g].Dot(weight3[g], grad3[g], true, false, true);
      }
    }
    Col2Img(gbottom, col_grad_);
  } else {
    Range nrng=grad_.IndexRange(0);
    for (int n = nrng.first; n < nrng.second; n++) {
      DAry grad3=grad4[n];
      DAry col3=col4[n];
      for (int g = 0; g < ngroups_; g++){
        gweight3[g].Dot(grad3[g], col3[g], false, true, false);
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
  float scale=1.0/(1.0-drop_prob_);
  gbottom->Map([scale](float g, float m) {return g*m*scale;}, grad_, mask_);
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
  vector<Range> slice{nrng, crng, Range({0, height_}),Range({0,width_})};
  const DAry& bottom=in_edges_[0]->GetData(this);
  const DAry& lbottom=bottom.Fetch(slice);
  switch (pooling_method_) {
    case LayerProto::kMaxPooling:
      data_.Fill(-FLT_MAX);
      for (int n = nrng.first; n < nrng.second; n++) {
        for (int c = crng.first; c < crng.second; c++) {
          // data should be continous for h and w dimension
          float* dptr=data_.addr(n,c,0,0);
          float* bdptr=lbottom.addr(n,c,0,0);
          for (int ph = 0; ph < pheight_; ph++) {
            for (int pw = 0; pw < pwidth_; pw++) {
              int hstart = ph * stride_;
              int wstart = pw * stride_;
              int hend = std::min(hstart + wsize_, height_);
              int wend = std::min(wstart + wsize_, width_);
              for (int h = hstart; h < hend; h++) {
                for (int w = wstart; w < wend; w++) {
                  *dptr= std::max(*dptr, bdptr[h*width_+w]);
                }
              }
              dptr++;
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
  vector<Range> slice{nrng, crng, Range({0, pheight_}),Range({0,pwidth_})};
  const DAry& ldata=data_.Fetch(slice);
  const DAry& lgrad=grad_.Fetch(slice);
  switch (pooling_method_) {
    case LayerProto::kMaxPooling:
      gbottom->Fill(0.0f);
      for (int n = nrng.first; n < nrng.second; n++) {
        for (int c = crng.first; c < crng.second; c++) {
          float *dptr=ldata.addr(n,c,0,0);
          float *gptr=lgrad.addr(n,c,0,0);
          float *bdptr=bottom.addr(n,c,0,0);
          float *bgptr=gbottom->addr(n,c,0,0);
          for (int ph = 0; ph < pheight_; ph++) {
            for (int pw = 0; pw < pwidth_; pw++) {
              int hstart = ph * stride_;
              int wstart = pw * stride_;
              int hend = std::min(hstart + wsize_, height_);
              int wend = std::min(wstart + wsize_, width_);
              for (int h = hstart; h < hend; h++) {
                for (int w = wstart; w < wend; w++) {
                  bgptr[h*width_+w]+=(*gptr)*(bdptr[h*width_+w]==(*dptr));
                  /*
                  gbottom->at(n,c,h,w) += lgrad.at(n,c,ph,pw)* (
                      bottom.at(n,c,h,w)==ldata.at(n,c,ph,pw));
                      */
                }
              }
              dptr++;
              gptr++;
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
  Range crng=data_.IndexRange(1);
  // crng.first<- max(0, data_.IndexRange(1).first-lpad_)
  //Range crng({0, data_.shape(1)});
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
    norm3[crng.first].Sum(squared3, Range(crng.first, crng.first+rpad_));
    for(int c=crng.first+1;c<crng.second;++c){
      DAry cur=norm3[c];
      cur.Copy(norm3[c-1]);
      if(c-lpad_>=crng.first)
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
  Range crng=bottom.IndexRange(1);
  // current crng is full
  //Range crng({0, bottom.shape(1)});
  std::vector<Range>slice{nrng, crng, Range({0, bottom.shape(2)}),Range({0, bottom.shape(3)})};
  const DAry& ldata=data_.Fetch(slice);
  const DAry& lgrad=grad_.Fetch(slice);
  const DAry& lnorm=norm_.Fetch(slice);
  gbottom->Pow(lnorm, -beta_);
  gbottom->Mult(*gbottom, lgrad);
  ratio_.Mult(lgrad, ldata);
  ratio_.Div(ratio_, lnorm);
  DAry accum_ratio(ratio_.shape().SubShape().SubShape());
  for(int n=nrng.first;n<nrng.second;++n) {
    DAry gbottom3=(*gbottom)[n];
    DAry bottom3=bottom[n];
    DAry ratio3=ratio_[n];
    accum_ratio.Sum(ratio3, {crng.first, lpad_});
    for(int c=crng.first;c<crng.second;++c) {
      if(c+lpad_<crng.second) accum_ratio.Add(ratio3[c+lpad_]);
      gbottom3[c].Map([factor](float g, float a, float b)
                      {return g+factor*a*b;}, gbottom3[c], accum_ratio, bottom3[c]);
      if(c-rpad_+1>=crng.first) accum_ratio.Minus(ratio3[c-rpad_+1]);
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
  if(pdim==1){
    weight_.SetupDAry(1);
    bias_.SetupDAry(0);
  }else if(pdim==0){
    weight_.SetupDAry(0);
    bias_.SetupDAry(-1);
  }else{
    weight_.SetupDAry(-1);
    bias_.SetupDAry(-1);
  }
}
void FCLayer::SetPartition(int pdim){
  Layer::SetPartition(pdim);
  if(pdim==1){
    weight_.SetPartition(1);
    bias_.SetPartition(0);
  }else if(pdim==0){
    weight_.SetPartition(0);
    bias_.SetPartition(-1);
  }else{
    weight_.SetPartition(-1);
    bias_.SetPartition(-1);
  }
}
bool FCLayer::PreSyncF(){
  const DAry& bottom=in_edges_[0]->GetData(this);
  if(bottom.GetPartition()==0&&data_.GetPartition()==0)
    return false;
  else
    return true;
}
bool FCLayer::PreSyncG(){
  return PreSyncF();
}

bool FCLayer::PostSyncF(){
  return PreSyncF();
}

bool FCLayer::PostSyncG(){
  return PreSyncF();
}
void FCLayer::ComputeFeature() {
  const DAry& bottom=in_edges_[0]->GetData(this);
  DAry bottom2=bottom.Reshape({num_, vdim_});
  vector<Range> slice{Range({0, num_}), Range({0, vdim_})};
  //cache_data_=bottom2.Fetch(slice);
  data_.Dot(bottom2, weight_.data());
  data_.AddRow(bias_.data());
  DAry* gbottom=in_edges_[0]->GetMutableGrad(this);
  gbottom->Fill(0.0f);
}

void FCLayer::ComputeGradient() {
  const DAry& bottom=in_edges_[0]->GetData(this);
  DAry* gbottom=in_edges_[0]->GetMutableGrad(this);
  DAry bottom2=bottom.Reshape({num_, vdim_});
  DAry grad2=grad_.Reshape({num_, hdim_});

  weight_.mutable_grad()->Dot(bottom2, grad2, true, false, true);
  //weight_.mutable_grad()->Dot(cache_data_, grad2, true, false, true);
  bias_.mutable_grad()->SumRow(grad2, true);

  // if dest_grad is nullptr, then we only compute gradients for parameters
  // this may happen when the lower layer is DataLayer
  // if (gbottom != nullptr) {
  DAry gbottom2=gbottom->Reshape({num_, vdim_});
  gbottom2.Dot(grad2, weight_.data(), false, true,true);
  data_.Fill(0.0f);
  //}
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
  if(pdim==0)
    Layer::SetPartition(0);
  else if(pdim==-1){
    Layer::SetPartition(-1);
  }else{
    LOG(ERROR)<<"Not supported parition dim for softmax layer "<<pdim<<"-th";
  }
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
  vector<Range> slice{nrng, {0,dim_}};
  DAry lbottom=bottom.Fetch(slice);
  for (int n = nrng.first; n < nrng.second; ++n) {
    DAry lbottom1=lbottom[n];
    float mmax = lbottom1.Max();
    DAry data=data_[n];
    data.Map([mmax](float v){return std::exp(v-mmax);}, lbottom1);
    float sum=data.Sum();
    data.Div(data, sum);
  }
}

void SoftmaxLossLayer::ComputeGradient() {
  const DAry& label=in_edges_[1]->GetData(this);
  DAry* gbottom=in_edges_[0]->GetMutableGrad(this);
  gbottom->Copy(data_);
  Range nrng=gbottom->IndexRange(0);
  DAry gbottom2=gbottom->Reshape({num_, dim_});
  DAry llabel=label.Fetch({nrng, Range({0,1})});
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
  Range nrng=data_.IndexRange(0);
  const DAry& labelary=in_edges_[1]->GetData(this);
  const DAry& llabel=labelary.Fetch({nrng, Range({0,1})});
  for(int n=nrng.first;n<nrng.second;n++) {
    int label=static_cast<int>(llabel.at(n,0));
    CHECK(label>=0&&label<1000)<<"label "<<label;
    float prob_of_truth=data_.at(n,label);
    if(accuracy){
      int nlarger=0;
      // count num of probs larger than the prob of the ground truth label
      for(int i=0;i<dim_;i++) {
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
  int nrecords=nrng.second-nrng.first;
  perf.set_precision(ncorrect*1.0/nrecords);
  perf.set_loss(logprob/nrecords);
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
  vector<int> shape=shapes[0];

  if(cropsize_>0){
    shape[2]=cropsize_;
    shape[3]=cropsize_;
  }
  data_.SetShape(shape);
  grad_.SetShape(shape);
}
void ImageLayer::AddInputRecord(const Record &record){
  Range nrng=grad_.IndexRange(0);
  CHECK_LT(offset_, nrng.second-nrng.first);
  int n=offset_+nrng.first;
  const DAryProto& image=record.image();
  int channels=image.shape(0);
  int height=image.shape(1);
  int width=image.shape(2);
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
  RegisterCreateFunction("ConcatLayer", CreateLayer(ConcatLayer));
  RegisterCreateFunction("SplitLayer", CreateLayer(SplitLayer));
  RegisterCreateFunction("ImgColLayer", CreateLayer(ImgColLayer));
  RegisterCreateFunction("ConvProductLayer", CreateLayer(ConvProductLayer));
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
