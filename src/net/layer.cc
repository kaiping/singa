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
  drop_prob_=proto.drop_prob();
  type_=proto.type();
}

void Layer::ToProto(LayerProto *proto) {
  proto->set_name(name_);
}
void Layer::Setup(const char flag){
  if(drop_prob_>0) {
    in_edges_[0]->SetupTopBlob(AllocData(flag),&drop_fea_);
    in_edges_[0]->SetupTopBlob(AllocData(flag),&drop_grad_);
    in_edges_[0]->SetupTopBlob(AllocData(flag),&mask_);
    VLOG(2)<<name_<<" drop prob is "<<drop_prob_<<drop_fea_.tostring();
  }
}
void Layer::Forward() {
  VLOG(3)<<name_;
  for(auto * edge: in_edges_)
    edge->Forward(&data_, edge->OtherSide(this)->data());
}

void Layer::Backward() {
  VLOG(3)<<name_;
  for(auto* edge: out_edges_){
    Layer * layer=edge->OtherSize(this).grad();
    edge->Backward(&grad_, data_, layer->grad(), layer->data());
  }
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
  RegisterCreateFunction("DataLayer", CreateLayer(DataLayer));
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
