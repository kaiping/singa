#include <glog/logging.h>
#include <cmath>
#include <chrono>
#include "utils/param.h"
#include "mshadow/tensor.h"
#include "utils/singleton.h"
using namespace mshadow;
using std::vector;
using std::string;
namespace singa {
Param::Param(){
  owner_=this;
}
Param::~Param(){}
  /*
void Param::FromProto(const ParamProto &proto){
  if(proto.ary_size()>=1){
    data_.FromProto(proto.ary(0));
    if(proto.ary_size()>=2){
      grad_.FromProto(proto.ary(1));
      //if(proto.ary_size()>=3)
       // history_.FromProto(proto.ary(2));
    }
  }
  param_proto_=proto;
  param_proto_.clear_ary();
}

void Param::ToProto(ParamProto *proto, bool copyData) {
  proto->CopyFrom(param_proto_);
  DAryProto* data=proto->add_ary();
  data_.ToProto(data, copyData);
  DAryProto* grad=proto->add_ary();
  grad_.ToProto(grad, copyData);
  //DAryProto* history=proto->add_ary();
  //history_.ToProto(history, copyData);
}
*/
void Param::ParseSyncMsgFromWorker(zmsg_t* msg){

}
zmsg_t *Param::GenSyncMsgFromWorker(){
  zmsg_t* msg=zmsg_new();
  return msg;
}

void Param::ParseSyncMsgFromPS(zmsg_t* msg){

}

zmsg_t *Param::GenSyncMsgFromPS(){
  zmsg_t* msg=zmsg_new();
  return msg;
}

void Param::Setup(const ParamProto& proto, const vector<int>& shape){
  data_.Reshape(shape);
  grad_.Reshape(shape);
  history_.Reshape(shape);
  param_proto_=proto;
}

void Param::Init(){
  Tensor<cpu, 1> data(data_.mutable_cpu_data(), Shape1(data_.count()));
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  auto random=ASingleton<Random<cpu>>::Instance(seed);
  switch (param_proto_.init_method()) {
  case ParamProto::kConstant:
    data=param_proto_.value();
    break;
  case ParamProto::kUniform:
    random->SampleUniform(data, param_proto_.low(), param_proto_.high());
    if(param_proto_.value())
      data*= param_proto_.value();
    break;
  case ParamProto::kUniformSqrtFanIn:
    random->SampleUniform(data, param_proto_.low(), param_proto_.high());
    if(param_proto_.value())
      data*= param_proto_.value()/ sqrt(data_.shape()[0] / 3.0f);
    break;
  case ParamProto::kUniformSqrtFanInOut:
    random->SampleUniform(data, param_proto_.low(), param_proto_.high());
    if(param_proto_.value())
      data*= param_proto_.value()/ sqrt(data_.shape()[0] +data_.shape()[1]);
    break;
  case ParamProto::kGaussain:
    random->SampleGaussian(data, param_proto_.mean(), param_proto_.std());
    if(param_proto_.value())
      data*= param_proto_.value();
    break;
  case ParamProto::kGaussainSqrtFanIn:
    random->SampleGaussian(data, param_proto_.mean(), param_proto_.std());
    if(param_proto_.value())
      data*= param_proto_.value()/ sqrt(data_.shape()[0]);
    break;
  default:
    LOG(ERROR) << "Illegal parameter init method ";
    break;
  }
}
}  // namespace singa
