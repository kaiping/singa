// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-03 18:02

#include <google/protobuf/repeated_field.h>
#include <glog/logging.h>
#include "model/param.h"
using google::protobuf::RepeatedField;

namespace lapis {
void Param::Init(const ParamProto &proto) {
  const RepeatedField<int> shape = proto.shape();
  momentum_=proto.momentum();
  learning_rate_=proto.learning_rate();
  weight_decay_=proto.weight_decay();
  // currently only support vector and  matrix parameter
  if (shape.size() == 2) {
    int h = shape.Get(0);
    int w = shape.Get(1);
    content_.Resize(Shape2(w, h));
    grad_.Resize(Shape2(w, h));
    history_grad_.Resize(Shape2(w, h));
  } else {
    int len = shape.Get(0);
    content_.Resize(Shape1(len));
    grad_.Resize(Shape1(l));
    history_.Resize(Shape1(len));
  }

  switch (proto.init_method()) {
    case ParamProto::kConstant:
      content_=proto.value();
      break;
    case ParamProto::kUniform:
      FillUniformData(proto.low(), proto.high(), proto.value());
      break;
    case ParamProto::kUniformSqrtFanIn:
      CHECK_EQ(shape.size(),2);
      FillUniformData(proto.low(), proto.high(),
                      proto.value()/sqrt(shape.Get(0)/3.0f));
      break;
    case ParamProto::kUniformSqrtFanInOut:
      CHECK_EQ(shape.size(),2);
      FillUniformData(proto.low(), proto.high(),
                      proto.value()/sqrt(shape.Get(0)+shape.Get(1)));
    case ParamProto::kGaussain:
      FillGaussainData(proto.mean(), proto.std(), proto.value());
      break;
    case ParamProto::kGaussainSqrtFanIn:
      CHECK_EQ(shape.size(),2);
      FillGaussainData(proto.mean(), proto.std(),
                       proto.value()/sqrt(shape.Get(0)));
      break;
    case ParamProto::kPretrained:
      break;
    default:
      LOG(ERROR)<<"Illegal parameter init method "<<proto.init_method();
  }
  name_ = proto.name();
}

void Param::ToProto(ParamProto *proto) {
  // TODO(wangwei) store the proto as a member for easy ToProto.
  proto->set_name(name_);
  proto->set_momentum(momentum_);
  proto->set_learning_rate(learning_rate_);
  proto->set_weight_decay(weight_decay_);
}

void Param::FillGaussainData(float mean, float std, float factor) {
  Random& rnd=Lapis::Instance()->rnd();
  rnd.SampleGaussian(content_, mean, std);
  if(factor!=1.0f)
    content_*=factor;
}

void Param::FillUniformData(float low, float high, float factor){
  LOG(INFO)<<low<<" "<<high;
  Random& rnd=Lapis::Instance()->rnd();
  rnd.SampleUniform(content_,low, high);
  if(factor!=1.0f)
    content_*=factor;
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
