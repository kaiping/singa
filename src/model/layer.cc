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

void Layer::Dropout(float drop_prob, float scale,
                    const Blob &src, Blob* dest, int * mask) {
  std::shared_ptr<std::mt19937> generator=Lapis::Instance()->generator();
  // with 1-drop_prob to keep one neuron, i.e., mask=1
  std::bernoulli_distribution distribution(1-drop_prob);
  float *dest_data=dest->mutable_data();
  const float* src_data=src.data();
  for(int i=0;i<src.length();i++) {
    mask[i]=distribution(*generator);
    dest_data[i]=src_data[i]*mask[i]*scale;
  }
}

void Layer::ComputeDropoutGradient(float scale, const Blob& src ,
                            const int* mask, Blob* dest) {
  const float* src_grad=src.data();
  float* dest_grad=dest->mutable_data();
  for(int i=0;i<dest->length();i++)
    dest_grad[i]=src_grad[i]*mask[i]*scale;
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
LayerFactory *LayerFactory::Instance() {
  static LayerFactory factory;
  return &factory;
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
