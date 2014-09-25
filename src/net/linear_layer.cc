// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-23 16:24

#include <glog/logging.h>
#include "net/linear_layer.h"

namespace lapis {
void LinearLayer::Setup(const char flag) {
  CHECK(in_edges_.size() == 1);
  in_edges_[0]->SetupTopBlob( AllocData(flag),&fea_);
  in_edges_[0]->SetupTopBlob( AllocData(flag),&grad_);
  VLOG(2)<<name_<<" shape: "<< fea_.tostring();
}

void LinearLayer::Forward() {
  Edge *edge = in_edges_[0];
  edge->Forward(edge->OtherSide(this)->feature(edge), &fea_, true);
  VLOG(3)<<"forwar linear layer "<<name_;
}

void LinearLayer::Backward() {
  Edge *edge = out_edges_[0];
  Layer *top = edge->OtherSide(this);
  edge->Backward(top->feature(edge), top->gradient(edge), fea_, &grad_, true);
  VLOG(3)<<"backward linear layer "<<name_;
}
}  // namespace lapis

