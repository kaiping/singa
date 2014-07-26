// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-23 16:24

#include <glog/logging.h>
#include "model/linear_layer.h"

namespace lapis {
void LinearLayer::Setup(int batchsize, TrainerProto::Algorithm alg,
                     const std::vector<DataSource *> &sources) {
  CHECK(in_edges_.size()==1);
  in_edges_[0]->SetupTopBlob(&fea_);
  in_edges_[0]->SetupTopBlob(&grad_);
}

void LinearLayer::Forward() {
  Edge* edge= in_edges_[0];
  edge->Forward(edge->bottom()->feature(edge), &fea_, true);
}

void LinearLayer::Backward(){
  Edge* edge= out_edges_[0];
  Layer* top=edge->top();
  edge->Backward(top->feature(edge),top->gradient(edge), &fea_, &grad_, true);
}
}  // namespace lapis

