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
    content_.Reshape(h, w);
    grad_.Reshape(h, w);
    history_grad_.Reshape(h, w);
  } else {
    int l = shape.Get(0);
    content_.Reshape(l);
    grad_.Reshape(l);
    history_grad_.Reshape(l);
  }

  int len=content_.length();
  float *val=content_.mutable_data();
  switch (proto.init_method()) {
    case ParamProto::kConstant: {
      for(int i=0;i<len;i++)
        val[i]=proto.value();
      break;
    }
    case ParamProto::kUniform: {
      FillUniformData(len, proto.low(), proto.high(), proto.value(), val);
      break;
    }
    case ParamProto::kUniformSqrtFanIn:{
      CHECK_EQ(shape.size(),2);
      FillUniformData(len, proto.low(), proto.high(),
                      proto.value()/sqrt(shape.Get(0)/3.0f), val);
      break;
    }
    case ParamProto::kUniformSqrtFanInOut: {
      CHECK_EQ(shape.size(),2);
      FillUniformData(len, proto.low(), proto.high(),
                      proto.value()/sqrt(shape.Get(0)+shape.Get(1)), val);
    }
    case ParamProto::kGaussain: {
      FillGaussainData(len, proto.mean(), proto.std(), proto.value(), val);
      break;
    }
    case ParamProto::kGaussainSqrtFanIn: {
      CHECK_EQ(shape.size(),2);
      FillGaussainData(len, proto.mean(), proto.std(),
                       proto.value()/sqrt(shape.Get(0)), val);
      break;
    }
    case ParamProto::kPretrained: {
      content_.set_data(proto.content().data());
      history_grad_.set_data(proto.history().data());
      break;
    }
    default: {
      LOG(ERROR)<<"Illegal parameter init method "<<proto.init_method();
    }
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

void Param::FillGaussainData(int length, float mean, float std, float factor, float *val) {
  std::normal_distribution<float> normal(mean,std);
  std::shared_ptr<std::mt19937> generator=Lapis::Instance()->generator();
  for (int i=0;i<length;i++)
    val[i]=normal(*generator)*factor;
}

void Param::FillUniformData(int length, float low, float high, float factor, float *val) {
  LOG(INFO)<<low<<" "<<high;
  std::shared_ptr<std::mt19937> generator=Lapis::Instance()->generator();
  std::uniform_real_distribution<float> uniform(low,high);
  for (int i=0;i<length;i++)
    val[i]=uniform(*generator)*factor;
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
