// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-14 11:42

#include <glog/logging.h>
#include "model/edge.h"
#include "model/sgd_trainer.h"
#include "model/lapis.h"
#include "model/conv_edge.h"
#include "model/inner_product_edge.h"
#include "model/lrn_edge.h"
#include "model/pooling_edge.h"
#include "model/softmax_loss_edge.h"


namespace lapis {
void Edge::Init(const EdgeProto &proto,
                const std::map<std::string, Layer *> &layer_map) {
  name_ = proto.name();
  type_=proto.type();
  CHECK(layer_map.find(proto.top()) != layer_map.end())<<proto.top();
  CHECK(layer_map.find(proto.bottom()) != layer_map.end())<<proto.bottom();
  top_ = layer_map.at(proto.top());
  bottom_ = layer_map.at(proto.bottom());
  if (proto.directed()) {
    top_->add_in_edge(this);
    bottom_->add_out_edge(this);
  } else {
    top_->add_out_edge(this);
    bottom_->add_out_edge(this);
    top_->add_in_edge(this);
    bottom_->add_in_edge(this);
  }
}

void Edge::ToProto(EdgeProto *proto) {
  proto->set_name(name_);
}

void Edge::Setup(bool set_param) {
  DLOG(INFO) << name_<<" does not implemente Setup func";
}

void Edge::SetupTopBlob(Blob* blob) {
  VLOG(1)<<"Edge "<<name_<<" does not implement SetupTopBlob";
}

void Edge::ComputeParamUpdates(const Trainer *trainer) {
  const SGDTrainer *sgd = reinterpret_cast<const SGDTrainer *> (trainer);
  float momentum = sgd->momentum();
  float weight_decay = sgd->weight_decay();
  float learning_rate = sgd->learning_rate();
  for (Param *param : params_) {
    int len = param->length();
    Tensor1 history(param->mutable_history().dptr, Shape1(len));
    const Tensor1 gradient(param->gradient().dptr, Shape1(len));
    const Tensor1 data(param->content().dptr, Shape1(len));
    momentum *= param->momentum();
    weight_decay *= param->weight_decay();
    learning_rate *= param->learning_rate();
    history = history * momentum - (gradient + weight_decay * data) * learning_rate;
  }
}

/*****************************************************************************
 * Edge Factory Implementation
 *****************************************************************************/
#define CreateEdge(EdgeClass) [](void)->Edge* {return new EdgeClass();}
std::shared_ptr<EdgeFactory> EdgeFactory::instance_;
std::shared_ptr<EdgeFactory> EdgeFactory::Instance() {
  if (!instance_.get()) {
    instance_.reset(new EdgeFactory());
  }
  return instance_;
}

EdgeFactory::EdgeFactory() {
  RegisterCreateFunction("ConvEdge", CreateEdge(ConvEdge));
  RegisterCreateFunction("InnerProductEdge", CreateEdge(InnerProductEdge));
  RegisterCreateFunction("LRNEdge", CreateEdge(LRNEdge));
  RegisterCreateFunction("PoolingEdge", CreateEdge(PoolingEdge));
  RegisterCreateFunction("SoftmaxLossEdge", CreateEdge(SoftmaxLossEdge));
}

void EdgeFactory::RegisterCreateFunction(
  const std::string id,
  std::function<Edge*(void)> create_function) {
  edge_map_[id] = create_function;
}

Edge *EdgeFactory::Create(const std::string id) {
  CHECK(edge_map_.find(id) != edge_map_.end())
      << "The initialization function " << id << " has not been registered";
  return edge_map_[id]();
}
}  // namespace lapis
