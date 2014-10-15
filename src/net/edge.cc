// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-14 11:42

#include <glog/logging.h>
#include "net/edge.h"

namespace lapis {
  /*
void Edge::Init(const EdgeProto &proto,
    const std::map<std::string, Layer *> &layers) {
  name_ = proto.name();
  type_=proto.type();
  is_directed_=proto.is_directed();
  CHECK(layers.find(proto.node1()) != layers.end())<<proto.node1()<<" not exist";
  CHECK(layers.find(proto.node2()) != layers.end())<<proto.node2()<<" not exist";
  node1_ = layers.at(proto.node1());
  node2_= layers.at(proto.node1());
  if (is_directed_) {
    node2_->add_in_edge(this);
    node1_->add_out_edge(this);
  } else {
    node1_->add_in_edge(this);
    node2_->add_in_edge(this);
  }
}
*/
const DAry& Edge::GetData(Layer* tolayer){
  if(tolayer==node1_)
    return node2_->GetData(this);
  else
    return node1_->GetData(this);
}
DAry*Edge:: GetMutableData(Layer* tolayer){
  if(tolayer==node1_)
    return node2_->GetMutableData(this);
  else
    return node1_->GetMutableData(this);
}
const DAry& Edge::GetGrad(Layer* tolayer){
  if(tolayer==node1_)
    return node2_->GetGrad(this);
  else
    return node1_->GetGrad(this);
}
DAry* Edge::GetMutableGrad(Layer* tolayer){
  if(tolayer==node1_)
    return node2_->GetMutableGrad(this);
  else
    return node1_->GetMutableGrad(this);
}

}  // namespace lapis
