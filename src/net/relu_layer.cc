// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-23 15:29

#include <glog/logging.h>
#include "utils/common.h"
#include "net/relu_layer.h"

namespace lapis {

const std::string type = "ReLULayer";

void ReLULayer::Setup(const char flag){
  Layer::Setup(flag);
  CHECK(in_edges_.size() == 1);
  in_edges_[0]->SetupTopBlob( AllocData(flag),&fea_);
  in_edges_[0]->SetupTopBlob(AllocData(flag),&fea_grad_);
  in_edges_[0]->SetupTopBlob( AllocData(flag),&act_);
  in_edges_[0]->SetupTopBlob( AllocData(flag),&act_grad_);
  VLOG(2)<<name_<<" Shape "<<fea_.tostring();
}

void ReLULayer::Forward() {
  Edge *edge = in_edges_[0];
  edge->Forward(edge->OtherSide(this)->feature(edge), &act_, true);
  VLOG(1)<<"forward relu";
  float *act = act_.dptr;
  if(drop_prob_>0) {
    float *drop_fea = drop_fea_.dptr;
    for (int i = 0; i < act_.length(); i++)
      drop_fea[i] = std::max(act[i], 0.0f);
    Dropout(drop_prob_, drop_fea_, &fea_, &mask_);
  }else {
    float *fea = fea_.dptr;
    for (int i = 0; i < act_.length(); i++)
      fea[i] = std::max(act[i], 0.0f);
  }
}

void ReLULayer::Backward() {
  Edge *edge = out_edges_[0];
  Layer *layer = edge->OtherSide(this);
  edge->Backward(layer->feature(edge), layer->gradient(edge), fea_, &fea_grad_,
                 true);
  VLOG(1)<<"backward relu";
  float *act_grad=act_grad_.dptr;
  // don't do dropout for test/validation
  if(drop_prob_>0){
    ComputeDropoutGradient(drop_prob_, fea_grad_, mask_, &drop_grad_);
    float *drop_grad=drop_grad_.dptr;
    float *drop_fea=drop_fea_.dptr;
    for (int i = 0; i < act_grad_.length(); i++)
      act_grad[i] = drop_grad[i] * (drop_fea[i] > 0);
  } else {
    float *fea_grad = fea_grad_.dptr;
    float *fea = fea_.dptr;
    for (int i = 0; i < act_grad_.length(); i++)
      act_grad[i] = fea_grad[i] * (fea[i] > 0);
  }
  VLOG(1)<<act_grad_.Norm();
}
}  // namespace lapis
