// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-03 18:02

#include <google/protobuf/repeated_field.h>
#include <glog/logging.h>
#include "model/param.h"
#include "utils/common.h"
using google::protobuf::RepeatedField;

namespace lapis {
void Param::Init(const ParamProto &proto, const char flag) {
  const RepeatedField<int>& shape = proto.shape();
  momentum_ = proto.momentum_multiplier();
  learning_rate_ = proto.learning_rate_multiplier();
  weight_decay_ = proto.weight_decay_multiplier();
  // currently only support vector and  matrix parameter
  if (shape.size() == 2) {
    int h = shape.Get(0);
    int w = shape.Get(1);
    content_.Resize(1,1,h,w, AllocParam(flag));
    grad_.Resize(1,1,h,w, AllocParam(flag));
    history_grad_.Resize(1,1,h,w, AllocParam(flag));
    VLOG(2)<<"Weight shape "<<content_.tostring();
  } else {
    int len = shape.Get(0);
    content_.Resize(1,1,1,len, AllocParam(flag));
    grad_.Resize(1,1,1,len, AllocParam(flag));
    history_grad_.Resize(1,1,1,len, AllocParam(flag));
    VLOG(2)<<"Bias shape "<<content_.tostring();
  }
  if(InitParam(flag)){
      switch (proto.init_method()) {
      case ParamProto::kConstant:
      for (int i = 0; i < content_.length(); i++)
      content_.dptr[i] = proto.value();
      break;
      case ParamProto::kUniform:
      FillUniformData(proto.low(), proto.high(), proto.value());
      break;
      case ParamProto::kUniformSqrtFanIn:
      CHECK_EQ(shape.size(), 2);
      FillUniformData(proto.low(), proto.high(),
        proto.value() / sqrt(shape.Get(0) / 3.0f));
      break;
      case ParamProto::kUniformSqrtFanInOut:
      CHECK_EQ(shape.size(), 2);
      FillUniformData(proto.low(), proto.high(),
        proto.value() / sqrt(shape.Get(0) + shape.Get(1)));
      break;
      case ParamProto::kGaussain:
      FillGaussainData(proto.mean(), proto.std(), proto.value());
      break;
      case ParamProto::kGaussainSqrtFanIn:
      CHECK_EQ(shape.size(), 2);
      FillGaussainData(proto.mean(), proto.std(),
          proto.value() / sqrt(shape.Get(0)));
      break;
      case ParamProto::kPretrained:
      LOG(ERROR)<<"Not implemented yet";
      break;
      default:
      LOG(ERROR) << "Illegal parameter init method " << proto.init_method();
      break;
      }
  }
  name_ = proto.name();
}

void Param::ToProto(ParamProto *proto) {
  // TODO(wangwei) store the proto as a member for easy ToProto.
  proto->set_name(name_);
  proto->set_momentum_multiplier(momentum_);
  proto->set_learning_rate_multiplier(learning_rate_);
  proto->set_weight_decay_multiplier(weight_decay_);
}

void Param::FillGaussainData(float mean, float std, float factor) {
  Random &rnd = Lapis::Instance()->rnd();
  Tensor1 content(content_.dptr, Shape1(length()));
  rnd.SampleGaussian(content, mean, std);
  if (factor != 1.0f)
    content *= factor;
}

void Param::FillUniformData(float low, float high, float factor) {
  Random &rnd = Lapis::Instance()->rnd();
  Tensor1 content(content_.dptr, Shape1(length()));
  rnd.SampleUniform(content, low, high);
  if (factor != 1.0f)
    content *= factor;
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
