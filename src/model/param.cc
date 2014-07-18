// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-03 18:02

#include <google/protobuf/repeated_field.h>
#include <glog/logging.h>
#include "model/param.h"
using google::protobuf::RepeatedField;

namespace lapis {

void Param::Init(const ParamProto &param_proto) {
  const RepeatedField<int> shape = param_proto.shape();
  if (shape.size() == 2) {
    int h = shape.Get(0);
    int w = shape.Get(1);
    content_.Reshape(h, w);
    grad_.Reshape(h, w);
    history_grad_.Reshape(h, w);
  } else {
    int l = shape.Get(0);
    content_.Reshape(l);
    grad_.Reshape(l);
    history_grad_.Reshape(l);
  }
  name_ = param_proto.name();

  // initialize the parameter
  ParamInitFactory::Instance()->Get(param_proto.initializer())(this);
}

/**************************************************************************
 * Implementation for ParamInitFactory
 *************************************************************************/

ParamInitFactory *ParamInitFactory::Instance() {
  static ParamInitFactory factory;
  return &factory;
}

void ParamInitFactory::RegisterInitFunc(
  std::string id, const std::function<void(Param *)> &func) {
  map_[id] = func;
}

std::function<void(Param *)> &ParamInitFactory::Get(std::string id) {
  CHECK(map_.find(id) != map_.end()) << "The initialization function " << id
                                     << " has not been registered\n";
  return map_[id];
}
}  // namespace lapis
