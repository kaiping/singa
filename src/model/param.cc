#include <glog/logging.h>
#include <cmath>
#include "model/param.h"

namespace singa {
void Param::FromProto(const ParamProto &proto){
  if(proto.ary_size()>=1){
    data_.FromProto(proto.ary(0));
    if(proto.ary_size()>=2){
      grad_.FromProto(proto.ary(1));
      if(proto.ary_size()>=3)
        history_.FromProto(proto.ary(2));
    }
  }
  /*
  if(proto.partition()!=-1&&GlobalContext::Get()->num_groups()==1)
    history_.InitFromProto(proto.data());
  */
  param_proto_=proto;
  param_proto_.clear_ary();
}

void Param::ToProto(ParamProto *proto, bool copyData) {
  proto->CopyFrom(param_proto_);
  DArrayProto* data=proto->add_ary();
  data_.ToProto(data, copyData);
  DArrayProto* grad=proto->add_ary();
  grad_.ToProto(grad, copyData);
  DArrayProto* history=proto->add_ary();
  history_.ToProto(history, copyData);
}

void Param::Setup(const vector<int>& shape, int partition_dim){
  data_.Setup(shape, partition_dim);
  grad_.Setup(shape, partition_dim);
  history_.Setup(shape, partition_dim);
  param_proto_.set_partition_dim(partition_dim);
}

void Param::Init(){
  CHECK(data_.shape().size)<<"must set shape of param";
  switch (param_proto_.init_method()) {
  case ParamProto::kConstant:
    data_.Fill(param_proto_.value());
    break;
  case ParamProto::kUniform:
    FillUniformData(param_proto_.low(), param_proto_.high(),
        param_proto_.value());
    break;
  case ParamProto::kUniformSqrtFanIn:
    FillUniformData(param_proto_.low(), param_proto_.high(),
        param_proto_.value() / sqrt(data_.shape(0) / 3.0f));
    break;
  case ParamProto::kUniformSqrtFanInOut:
    FillUniformData(param_proto_.low(), param_proto_.high(),
        param_proto_.value() / sqrt(data_.shape(0) + data_.shape(1)));
    break;
  case ParamProto::kGaussain:
    FillGaussainData(param_proto_.mean(), param_proto_.std(),
        param_proto_.value());
    break;
  case ParamProto::kGaussainSqrtFanIn:
    FillGaussainData(param_proto_.mean(), param_proto_.std(),
        param_proto_.value() / sqrt(data_.shape(0)));
    break;
  default:
    LOG(ERROR) << "Illegal parameter init method ";
    break;
  }
}

void Param::FillGaussainData(float mean, float std, float factor) {
  data_.SampleGaussian(mean, std);
  if (factor != 1.0f)
    data_.Mult(data_,factor);
}

void Param::FillUniformData(float low, float high, float factor) {
  data_.SampleUniform(low, high);
  if (factor != 1.0f)
    data_.Mult(data_,factor);
}
}  // namespace singa
