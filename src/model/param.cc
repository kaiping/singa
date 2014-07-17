// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-03 18:02

#include <google/protobuf/repeated_field.h>
#include <glog/logging.h>
#include "model/param.h"
using google::protobuf::RepeatedField;

namespace lapis {

void Param::Init(const ParamProto &param_proto) {
  const RepeatedField<int> shape=param_proto.shape();
  if (shape.size()==2) {
    int h=shape.Get(0);
    int w=shape.Get(1);
    content_.Reshape(h,w);
    grad_.Reshape(h,w);
    history_grad_.Reshape(h,w);
  } else {
    int l=shape.Get(0);
    content_.Reshape(l);
    grad_.Reshape(l);
    history_grad_.Reshape(l);
  }
  name_ = param_proto.name();

  // initialize the parameter
  ParamInitFactory::Instance()->Get(param_proto.initializer())(&content_);
}

/**************************************************************************
 * Implementation for ParamInitFactory
 *************************************************************************/

ParamInitFactory* ParamInitFactory::Instance() {
  static ParamInitFactory factory;
  return &factory;
}

void ParamInitFactory::RegisterInitFunc(std::string type,
    std::function<void(Blob*)> &func){
  map_[type]=func;
}

std::function<void(Blob*)>& ParamInitFactory::Get(std::string type) {
  CHECK(map_.find(type)!=map_.end())<<"The initialization function "<<type
    <<" has not been registered\n";
  return map_[type];
}
}  // namespace lapis
