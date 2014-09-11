// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-15 11:11
#include <glog/logging.h>
#include "net/data_layer.h"
#include "net/trainer.h"

namespace lapis {

const std::string DataLayer::kType = "DataLayer";

void DataLayer::Init(const LayerProto &proto) {
  Layer::Init(proto);
  data_source_= proto.data_source();
  mirror_=proto.mirror();
  cropsize_=proto.cropsize();
}

void DataLayer::ToProto(LayerProto *layer_proto) {
  Layer::ToProto(layer_proto);
  layer_proto->set_data_source(data_source_);
}

void DataLayer::SetInputShape(int batchsize, const Shape &data_shape){
  batchsize_=batchsize;
  width_=data_shape.width();
  height_=data_shape.height();
  channels_=data_shape.channels();
}

void DataLayer::SetInputStore(int store_id) {
  store_id_=store_id;
}

void DataLayer::Setup(const char flag) {
  VLOG(2)<<"DataLayer: "<<name_<<" cropsize "<<cropsize_;
  if(cropsize_){
    VLOG(3)<<"crop size "<<cropsize_<<" alloc data "<<AllocData(flag);
    data_.Reshape(batchsize_, channels_,cropsize_, cropsize_, AllocData(flag));
    tmp_.Reshape(batchsize_,channels_,width_, height_, AllocData(flag));
    VLOG(2)<<" shape before crop "<<tmp_.tostring()
      <<" shape after crop "<<data_.tostring();
  }
  else {
    data_.Reshape(batchsize_,channels_, height_,width_,AllocData(flag));
    VLOG(2) <<" shape "<<data_.tostring();
  }
}

void DataLayer::LoadData(const DAry &input, Phase phase){
  VLOG(3)<<name_;
  if(cropsize_){
    CropImages(data_, input_, stage==kTrainStage, mirror_);
  }else{
    Copy(data_, input_);
  }
}

void DataLayer::Forward() {}
void DataLayer::Backward() {
  VLOG(3)<<name_;
  for (Edge *edge : out_edges_) {
    Layer *layer = edge->OtherSide(this);
    edge->Backward(layer->data(edge), layer->grad(edge), data_, nullptr, true);
  }
}
}  // namespace lapis
