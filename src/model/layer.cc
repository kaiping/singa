// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-06 15:19

#include <glog/logging.h>
#include <memory>
#include <random>
#include "mshadow/tensor.h"


#include "model/layer.h"
#include "model/data_layer.h"
#include "model/linear_layer.h"
#include "model/relu_layer.h"

namespace lapis {
/*****************************************************************************
 * Implementation for Layer
 *****************************************************************************/
void Layer::Init(const LayerProto &proto) {
  name_ = proto.name();
}

void Layer::ToProto(LayerProto *proto) {
  proto->set_name(name_);
}

void Layer::Dropout(float drop_prob, const Blob &src, Blob *dest, Blob *mask) {
  Random &rnd = Lapis::Instance()->rnd();
  // with 1-drop_prob to keep one neuron, i.e., mask=1
  float keep_prob = 1.0 - drop_prob;
  int len = dest->length();
  Tensor1 dest_t(dest->dptr, Shape1(len));
  Tensor1 mask_t(mask->dptr, Shape1(len));
  Tensor1 src_t(src.dptr, Shape1(len));
  mask_t = mshadow::expr::F<mshadow::op::threshold>(rnd.uniform(mask_t.shape),
           keep_prob);
  dest_t = src_t * mask_t *(1.0 / keep_prob);
}

void Layer::ComputeDropoutGradient(const Blob &src ,
                                   const Blob &mask, Blob *dest) {
  int len = dest->length();
  Tensor1 dest_t(dest->dptr, Shape1(len));
  Tensor1 mask_t(mask.dptr, Shape1(len));
  Tensor1 src_t(src.dptr, Shape1(len));
  dest_t = src_t * mask_t;
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
std::shared_ptr<LayerFactory> LayerFactory::Instance() {
  if (!instance_.get()) {
    instance_.reset(new  LayerFactory());
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
