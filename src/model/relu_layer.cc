// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-23 15:29

#include <glog/logging.h>

#include "model/relu_layer.h"

namespace lapis {

const std::string type="ReLULayer";

void ReLULayer::Setup(int batchsize, TrainerProto::Algorithm alg,
                      const std::vector<DataSource *> &sources) {
  CHECK(in_edges_.size()==1);
  in_edges_[0]->SetupTopBlob(&fea_);
  in_edges_[0]->SetupTopBlob(&fea_grad_);
  in_edges_[0]->SetupTopBlob(&act_);
  in_edges_[0]->SetupTopBlob(&act_grad_);
}

struct relu {
  inline static float Map(float a) {
    return std::max(a, 0.0f);
  }
};
struct relu_grad {
  inline static float Map(float a) {
    return a>0.0f ?1.0f:0.0f;
  }
};

void ReLULayer::Forward() {
  Edge* edge= in_edges_[0];
  edge->Forward(edge->OtherSide(this)->feature(edge), &act_, true);
  fea_data_=F<relu>(act_);
}

void ReLULayer::Backward() {
  Edge* edge= out_edges_[0];
  Layer* layer=edge->OtherSide(this);
  edge->Backward(layer->feature(edge),layer->gradient(edge), fea_, &fea_grad_, true);
  float* act_grad_data=act_grad_.mutable_data();
  act_grad_=fea_grad_*F<relu_grad>(act_);
}
}  // namespace lapis

