// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-03 18:02

#include <google/protobuf/repeated_field.h>
#include <glog/logging.h>
#include "net/param.h"
#include "utils/common.h"
using google::protobuf::RepeatedField;

namespace lapis {
void Param::Init(const ParamProto &proto){
  name_=proto.name();
  momentum_ = proto.momentum_multiplier();
  learning_rate_ = proto.learning_rate_multiplier();
  weight_decay_ = proto.weight_decay_multiplier();
  init_method_=proto.init_method();

  low_=low_;
  high_=high_;
  mean_=proto.mean();
  std_=proto.std();
  value_=value_;
}

void Param::SetShape(int len){
  data_.SetShape({len});
  grad_.SetShape({len});
}
void Param::SetShape(int h, int w){
  data_.SetShape({h,w});
  grad_.SetShape({h,w});
}

void Param::FreeMemory() {
  data_.FreeMemory();
  grad_.FreeMemory();
}

void Param::Fill(){
  CHECK(data_.Shape().size())<<"must set shape of param";
  if(!data_.allocated())
    data_.AllocateMemory();
  switch (init_method_) {
  case ParamProto::kConstant:
    data_.set(value_);
    break;
  case ParamProto::kUniform:
    FillUniformData(low_, high_, value_);
    break;
  case ParamProto::kUniformSqrtFanIn:
    CHECK_EQ(data_.Shape().size(), 2);
    FillUniformData(low_ , high_, value_ / sqrt(data_.Shape(0) / 3.0f));
    break;
  case ParamProto::kUniformSqrtFanInOut:
    CHECK_EQ(data_.Shape().size(), 2);
    FillUniformData(low_, high_, value_ / sqrt(data_.Shape(0) + data_.Shape(1)));
    break;
  case ParamProto::kGaussain:
    FillGaussainData(mean_ std_, value_);
    break;
  case ParamProto::kGaussainSqrtFanIn:
    CHECK_EQ(shape.size(), 2);
    FillGaussainData(mean_,std_, value_ / sqrt(data_.Shape(0)));
    break;
  case ParamProto::kPretrained:
    LOG(ERROR)<<"Not implemented yet";
    break;
  default:
    LOG(ERROR) << "Illegal parameter init method " << init_method_;
    break;
  }
}


void Param::ToProto(ParamProto *proto) {
  // TODO(wangwei) store the proto as a member for easy ToProto.
  proto->set_name(name_);
  proto->set_momentum_multiplier(momentum_);
  proto->set_learning_rate_multiplier(learning_rate_);
  proto->set_weight_decay_multiplier(weight_decay_);
}

void Param::FillGaussainData(float mean, float std, float factor) {
  DAry::SampleGaussian(&data_, mean, std);
  if (factor != 1.0f)
    DAry::Mult(&data_, data_,factor);
}

void Param::FillUniformData(float low, float high, float factor) {
  DAry::SampleUniform(&data_, low, high);
  if (factor != 1.0f)
    DAry::Mult(&data_, data_,factor);
}
}  // namespace lapis
