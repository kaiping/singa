// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-03 18:02

#include <glog/logging.h>
#include <cmath>

#include "net/param.h"
#include "utils/common.h"

namespace lapis {
void Param::Init(const ParamProto &proto){
  name_=proto.name();
  //momentum_multiplier_ = proto.momentum_multiplier();
  learning_rate_multiplier_ = proto.learning_rate_multiplier();
  weight_decay_multiplier_ = proto.weight_decay_multiplier();
  init_method_=proto.init_method();

  low_=proto.low();
  high_=proto.high();
  mean_=proto.mean();
  std_=proto.std();
  value_=proto.value();
  split_threshold_=proto.split_threshold();
  if(proto.has_data())
    data_.InitFromProto(proto.data());
  if(proto.has_grad())
    grad_.InitFromProto(proto.grad());
}

void Param::ToProto(ParamProto *proto, bool copyData) {
  // TODO(wangwei) store the proto as a member for easy ToProto.
  proto->set_name(name_);
  proto->set_learning_rate_multiplier(learning_rate_multiplier_);
  proto->set_weight_decay_multiplier(weight_decay_multiplier_);
  proto->set_init_method(init_method_);
  proto->set_split_threshold(split_threshold_);
  proto->set_mean(mean_);
  proto->set_std(std_);
  proto->set_low(low_);
  proto->set_high(high_);
  proto->set_value(value_);

  DAryProto* data=proto->mutable_data();
  data_.ToProto(data, copyData);
  DAryProto* grad=proto->mutable_grad();
  grad_.ToProto(grad, copyData);
}
void Param::SetShape(int len){
  data_.SetShape({len});
  grad_.SetShape({len});
}
void Param::SetShape(int h, int w){
  data_.SetShape({h,w});
  grad_.SetShape({h,w});
}
void Param::SetPartition(int k) {
  data_.SetPartition(k);
  grad_.SetPartition(k);
}
void Param::SetupDAry(int k) {
  data_.Setup(k);
  grad_.Setup(k);
}
void Param::Fill(){
  CHECK(data_.shape().size)<<"must set shape of param";
  switch (init_method_) {
  case ParamProto::kConstant:
    data_.Fill(value_);
    break;
  case ParamProto::kUniform:
    FillUniformData(low_, high_, value_);
    break;
  case ParamProto::kUniformSqrtFanIn:
    CHECK_EQ(data_.shape().size, 2);
    FillUniformData(low_ , high_, value_ / sqrt(data_.shape(0) / 3.0f));
    break;
  case ParamProto::kUniformSqrtFanInOut:
    CHECK_EQ(data_.shape().size, 2);
    FillUniformData(low_, high_, value_ / sqrt(data_.shape(0) + data_.shape(1)));
    break;
  case ParamProto::kGaussain:
    FillGaussainData(mean_, std_, value_);
    break;
  case ParamProto::kGaussainSqrtFanIn:
    CHECK_EQ(data_.shape().size, 2);
    FillGaussainData(mean_,std_, value_ / sqrt(data_.shape(0)));
    break;
  case ParamProto::kPretrained:
    LOG(ERROR)<<"Not implemented yet";
    break;
  default:
    LOG(ERROR) << "Illegal parameter init method " << init_method_;
    break;
  }
}



void Param::FillGaussainData(float mean, float std, float factor) {
  data_.SampleGaussian(mean, std);
  if (factor != 1.0f)
    data_.Mult( data_,factor);
}

void Param::FillUniformData(float low, float high, float factor) {
  data_.SampleUniform(low, high);
  if (factor != 1.0f)
    data_.Mult( data_,factor);
}
}  // namespace lapis
