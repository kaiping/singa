// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-06 15:19

#include <glog/logging.h>
#include <memory>
#include <random>

#include "model/layer.h"

namespace lapis {
/*****************************************************************************
 * Implementation for Layer
 *****************************************************************************/
void Layer::Init(const LayerProto &proto){
  name_ = proto.name();
}

void Layer::ToProto(LayerProto *proto) {
  proto->set_name(name_);
}
struct Threshold {
  inline static float Map(float a, float b) {
    return a<b?1.0f: 0.0f;
  }
};

void Layer::Dropout(float drop_prob, const Blob4 &src, Blob4* dest, Blob4 * mask) {
  Random& rnd=Lapis::Instance()->rnd();
  // with 1-drop_prob to keep one neuron, i.e., mask=1
  float keep_prob=1.0-drop_prob;
  *mask=F<Threashold>(rnd.uniform(mask->shape), keep_prob)*(1.0/keep_prob);
  *dest=(*mask)*src;
}

void Layer::ComputeDropoutGradient(const Blob4& src ,
                            const Blob4& mask, Blob4* dest) {
  *dest=src*mask;
}
// Currently layers do not have parameters
/*
void Layer::ComputeParamUpdates(const Trainer *trainer) {
  LOG(INFO) << "Layer " << name_ << " has no parameters to update\n";
}
*/

/*****************************************************************************
 * Implementation for LayerFactory
 ****************************************************************************/
#define CreateLayer(LayerClass) [](void)->Layer* {return new LayerClass();}
static std::shared_ptr<LayerFactory> LayerFactory::Instance() {
  if(!instance_.get()) {
    instance.reset(new  LayerFactory());
  }
  return instance_;
}

LayerFactory::LayerFactory() {
  RegisterCreateFunction("DataLayer", CreateLayer(DataLayer));
  RegisterCreateFunction("LinearLayer", CreateLayer(LinearLayer));
  RegisterCreateFunction("ReLULayer", CreateLayer(ReLULayer));
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
