
#include "utils/param_updater.h"
#include "mshadow/tensor.h"
#include "proto/model.pb.h"
using namespace mshadow;
using namespace mshadow::expr;

namespace  singa {

float ParamUpdater::GetLearningRate(int step){
  float ret = 0., r = 0., base=proto_.base_learning_rate();
  int freq=0;
  switch (proto_.learning_rate_change_method()) {
    case UpdaterProto_ChangeProto_kFixed:
      ret = base;
      break;
    case UpdaterProto_ChangeProto_kLinear:
      // a is init, b is the final
      freq=proto_.learning_rate_change_frequency();
      r = step * 1.0  / freq;
      ret = (1.0 - r) * base + r * proto_.final_learning_rate();
      break;
    case UpdaterProto_ChangeProto_kExponential:
      // a is init, b is the final, from convnet
      CHECK_EQ(base, 2 * proto_.final_learning_rate())
        << "final value should be the half";
      freq=proto_.learning_rate_change_frequency();
      ret = base / pow(2, step * 1. / freq);
      break;
    case UpdaterProto_ChangeProto_kInverse_t:
      // a is init, b is the final, from convnet
      CHECK_EQ(base, 2 * proto_.final_learning_rate())
        << "final value should be the half";
      ret = base / (1. + step * 1. / proto_.final_learning_rate());
      break;
    case UpdaterProto_ChangeProto_kInverse:
      // a is init, b is gamma, c is pow
      ret=base*pow(1.f+proto_.gamma()*step, -proto_.pow());
      break;
    case UpdaterProto_ChangeProto_kStep:
      // a is the base learning rate, b is gamma, from caffe
      // notice it is step/change_steps, not step*1.0/change_steps
      freq=proto_.learning_rate_change_frequency();
      ret = base * pow(proto_.gamma(), step / freq);
      break;
    default:
      LOG(ERROR) << "Wrong hyper-parameter update method";
  }
  return ret;
}

void AdaGradUpdater::Init(const UpdaterProto& proto){
  ParamUpdater::Init(proto);
  base_lr_=proto.base_learning_rate();
  delta_=proto.delta();
  weight_decay_=proto.weight_decay();
}

void AdaGradUpdater::Update(int step, Param* param){
  Shape<1> s=Shape1(param->size());
  Tensor<cpu, 1> data(param->mutable_data()->mutable_cpu_data(), s);
  Tensor<cpu, 1> grad(param->mutable_grad()->mutable_cpu_data(), s);
  Tensor<cpu, 1> history(param->mutable_history()->mutable_cpu_data(), s);
  history+=F<op::square>(grad);
  float lr=GetLearningRate(step)*param->learning_rate_multiplier();
  float wd=weight_decay_*param->weight_decay_multiplier();
  if(wd>0){ // L2 regularization
    grad+=data*wd;
  }
  data-=lr*grad/(F<op::sqrtop>(history)+delta_);
}
} /* singa */
