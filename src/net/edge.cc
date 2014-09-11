// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-14 11:42

#include <glog/logging.h>
#include "net/edge.h"
#include "net/sgd_trainer.h"
#include "net/lapis.h"
#include "net/conv_edge.h"
#include "net/inner_product_edge.h"
#include "net/lrn_edge.h"
#include "net/pooling_edge.h"
#include "net/softmax_loss_edge.h"


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

void Edge::Setup(const char flag) {
  DLOG(INFO) << name_<<" does not implemente Setup func";
}

void Edge::SetupTop(const bool alloc, DAry* ary) {
  DLOG(INFO)<<"Edge "<<name_<<" does not implement SetupTopBlob";
}

void Edge::ComputeParamUpdates(const Trainer *trainer) {
  const SGDTrainer *sgd = reinterpret_cast<const SGDTrainer *> (trainer);
  float mom= sgd->momentum();
  float wdecay = sgd->weight_decay();
  float lr = sgd->learning_rate();
  for (Param *param : params_) {
    mom*= param->momentum();
    wdecay *= param->weight_decay();
    lr*= param->learning_rate();
    DAry* history=param->mutable_history();
    Mult(history, param->history, mom);
    Axpb(history, -lr, param->gradient(), param->history());
    Axpb(history, -lr*wdecay, param->data(), param->history());
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
  RegisterCreateFunction("DropoutEdge", CreateEdge(DropoutEdge));
  RegisterCreateFunction("ReLUEdge", CreateEdge(ReLUEdge));
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
