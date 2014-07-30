// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-23 15:29

#include <glog/logging.h>

#include "model/relu_layer.h"

namespace lapis {

const std::string type = "ReLULayer";

void ReLULayer::Setup(int batchsize, TrainerProto::Algorithm alg,
                      const std::vector<DataSource *> &sources) {
  CHECK(in_edges_.size() == 1);
  in_edges_[0]->SetupTopBlob(&fea_);
  in_edges_[0]->SetupTopBlob(&fea_grad_);
  in_edges_[0]->SetupTopBlob(&act_);
  in_edges_[0]->SetupTopBlob(&act_grad_);
}

void ReLULayer::Forward() {
  Edge *edge = in_edges_[0];
  edge->Forward(edge->OtherSide(this)->feature(edge), &act_, true);
  float *fea=fea_.dptr;
  float *act=act_.dptr;
  for(unsigned int i=0;i<act_.shape.Size();i++)
    fea[i]=std::max(act[i],0.0f);
}

void ReLULayer::Backward() {
  Edge *edge = out_edges_[0];
  Layer *layer = edge->OtherSide(this);
  edge->Backward(layer->feature(edge), layer->gradient(edge), fea_, &fea_grad_,
                 true);
  float *act_grad= act_grad_.dptr;
  float *fea_grad=fea_grad_.dptr;
  float *fea=fea_.dptr;
  for(unsigned int i=0;i<act_grad_.shape.Size();i++)
    act_grad[i]=fea_grad[i]*(fea[i]>0);
}
}  // namespace lapis

