// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-14 11:42

#include <glog/logging.h>
#include "model/edge.h"
#include "model/sgd_trainer.h"
#include "utils/lapis.h"


namespace lapis {
void Edge::Init(const EdgeProto &proto,
    const std::map<std::string, Layer *> &layer_map){
  name_ = proto.name();
  CHECK(layer_map.find(proto.top())!=layer_map.end());
  CHECK(layer_map.find(proto.bottom())!=layer_map.end());
  top_=layer_map.at(proto.top());
  bottom_=layer_map.at(proto.bottom());
  if(proto.directed()) {
      top_->add_in_edge(this);
      bottom_->add_out_edge(this);
  }else{
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
  LOG(INFO)<<"Not implemented";
}

void Edge::ComputeParamUpdates(const Trainer *trainer) {
  const SGDTrainer* sgd=reinterpret_cast<const SGDTrainer*> (trainer);
  float momentum=sgd->momentum();
  float weight_decay=sgd->weight_decay();
  float learning_rate=sgd->learning_rate();
  for (Param* param : params_) {
    AVec history(param->mutable_history(), param->length());
    AVec gradient(param->mutable_gradient(), param->length());
    AVec data(param->mutable_content(), param->length());
    momentum*=param->momentum();
    weight_decay*=param->weight_decay();
    learning_rate*=param->learning_rate();
    history=history*momentum-(gradient+weight_decay*data)*learning_rate;
  }
}
}  // namespace lapis
