// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-23 15:29

#include <glog/logging.h>

#include "model/relu_layer.h"

namespace lapis {

const std::string type="ReLULayer";

void ReLULayer::Setup(int batchsize, TrainAlgorithm alg,
                      const std::vector<DataSource *> &sources) {
  CHECK(in_edges_.size()==1);
  in_edges_[0]->SetupTopBlob(&fea_);
  in_edges_[0]->SetupTopBlob(&fea_grad_);
  in_edges_[0]->SetupTopBlob(&act_);
  in_edges_[0]->SetupTopBlob(&act_grad_);
}

void ReLULayer::Forward() {
  Edge* edge= in_edges_[0];
  edge->Forward(edge->OtherSide(this)->feature(edge), &act_, true);
  const float* act_data=act_.data();
  float* fea_data=fea_.mutable_data();
  for(int i=0;i<act_.length();i++)
    fea_data[i]=std::max(act_data[i],0.f);
}

void ReLULayer::Backward() {
  Edge* edge= out_edges_[0];
  Layer* layer=edge->OtherSide(this);
  edge->Backward(layer->feature(edge),layer->gradient(edge), &fea_, &fea_grad_, true);
  const float* fea_grad_data=fea_grad_.data();
  const float* act_data=act_.data();
  float* act_grad_data=act_grad_.mutable_data();
  for(int i=0;i<fea_.length();i++)
    act_grad_data[i]=fea_grad_data[i]*(act_data[i]>0);
}
}  // namespace lapis

