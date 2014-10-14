// Copyright © 2014 Jinyang Gao. All Rights Reserved.
// 2014-07-18 16:33

#include <vector>

#include "core/table_delegate.h"
#include "core/disk-table.h"
#include "core/table.h"
#include "proto/model.pb.h"

#include "darray/dary.h"

namespace lapis {
UpdateHandler<AdaGradValue>::UpdateHandler(const SolverProto& solver){
}
bool UpdateHandler<AdaGradValue>::Update(AdaGradValue* data, const AdaGradValue& update){
  /*
  Vector dst(data->grad());
  Vector src(update->grad());
  Add(*dst, dst, src);
  */
  return true;
}

UpdateHandler<SGDValue>::UpdateHandler(const SolverProto& solver){
  step_=0;
  base_learning_rate_=solver.sgd().base_learning_rate();
  gamma_=solver.sgd().gamma();
  learning_rate_change_=solver.sgd().learning_rate_change();
  learning_rate_change_steps_=solver.sgd().learning_rate_change_steps();
  momentum_=solver.sgd().momentum();
  weight_decay_=solver.sgd().weight_decay();
}
bool UpdateHandler<SGDValue>::Update(SGDValue* data, const SGDValue& update){
  if(update.version()!=data->version()){
    CHECK_EQ(data->version()+1, update.version());
    data->set_version(update.version());
    UpdateHyperParams(data->version());
  }
  int len=data->data().value_size();
  float* history=data->mutable_grad()->mutable_value()->mutable_data();
  float lr=learning_rate_*data->learning_rate_multiplier();
  float w=weight_decay_*data->weight_decay_multiplier();
  const float* grad=update.grad().value().data();
  float* dptr=data->mutable_data()->mutable_value()->mutable_data();
  // hist=hist-lr*grad
  DAry::arymath().madd(history, -lr, grad, history, len);
  // hist=hist-lr*weight*param
  DAry::arymath().madd(history, -lr*w, dptr, history, len);
  data->set_n_update(data->n_update()+1);

  if(data->n_update()==data->threshold()){
    // param+=history/n
    DAry::arymath().madd(dptr, 1.0f/data->n_update(), history, dptr, len);
    // hist=hist*mom
    DAry::arymath().mul(history, momentum_, history, len);
    data->set_n_update(0);
  }
  return true;
}
float UpdateHyperParam(int step, SGDValue::ChangeProto change, int change_steps, float a, float b) {
  float ret = 0., r = 0.;
  switch (change) {
    case SGDValue::kFixed:
      ret = a;
      break;
    case SGDValue::kLinear:
      // a is init, b is the final
      r = step * 1.0  / change_steps;
      ret = (1.0 - r) * a + r * b;
      break;
    case SGDValue::kExponential:
      // a is init, b is the final, from convnet
      CHECK_EQ(a, 2 * b) << "final value should be the half";
      ret = a / pow(2, step * 1. / change_steps);
      break;
    case SGDValue::kInverse_t:
      // a is init, b is the final, from convnet
      CHECK_EQ(a, 2 * b) << "final value should be the half";
      ret = a / (1. + step * 1. / b);
      break;
    case SGDValue::kStep:
      // a is the base learning rate, b is gamma, from caffe
      ret = a * pow(b, step / change_steps);
      break;
    default:
      LOG(ERROR) << "Wrong hyper-parameter update method";
  }
  return ret;
}
void UpdateHandler<SGDValue>::UpdateHyperParams(const int step) {
  learning_rate_ = UpdateHyperParam(step, learning_rate_change_,
      learning_rate_change_steps_,
      base_learning_rate_,
      gamma_);
  /*
  momentum_ = UpdateHyperParam(step, sgd_proto_.momentum_change(),
      sgd_proto_.momentum_change_steps(),
      sgd_proto_.base_momentum(),
      sgd_proto_.momentum_x());
  weight_decay_ = UpdateHyperParam(step, sgd_proto_.weight_decay_change(),
      sgd_proto_.weight_decay_change_steps(),
      sgd_proto_.base_weight_decay(),
      sgd_proto_.weight_decay_x());
      */
}

TypedTableDelegate::~TypedTableDelegate(){
  delete param_table_;
}

void TableDelegate::Update(const std::vector<Param*> &params, int step) {
  for(auto* param: params)
    Update(param,step);
  return;
}

void TableDelegate::Put(const std::vector<Param*> &params) {
  VLOG(3)<<"model controller put";
  if(GlobalContext::Get()->standalone())return;
  for(auto* param: params)
    Put(param);
}

void TableDelegate::Get(const std::vector<Param*> &params, int step){
  if(GlobalContext::Get()->standalone())return;
  for(auto* param : params)
    Get(param, step);
  return;
}
void TableDelegate::AsyncGet(const std::vector<Param*> &params, int step){
  if(GlobalContext::Get()->standalone())return;
  for(auto* param : params)
    AsyncGet(param, step);
  return;
}

template<>
void TypedTableDelegate<VKey, SGDValue>::SplitParams(const std::vector<Param *> &params, int wid) {

}
template<>
void TypedTableDelegate<VKey, AdaGrad>::SplitParams(const std::vector<Param *> &params, int wid) {

}

template<>
void TypedTableDelegate<int ,SGDValue>::Update(Param *param, int step){
  int paramid = param->id();
  int largestoffset = param->data().shape().Size();
  int splitsize = GlobalContext::Get()->num_table_servers()*split_size_;
  int splitoffset = largestoffset/splitsize;
  if (largestoffset%splitsize) splitoffset++;
  if (splitoffset > 1000000)
  {
    splitoffset = 1000000;
    splitsize = largestoffset/splitoffset + 1;
  }
  if (splitsize > 2048)VLOG(3)<<"Error:split size too much!!!";
  int curoffset = 0;

  const float * grad_addr = param->mutable_grad()->dptr();
  for(int j = 0; j < splitsize; j++)
  {

    SGDValue v;
    v.set_version(step);
    DAryProto* dary=v.mutable_grad();
    dary->clear_value();
    for(int k = 0; k < splitoffset; k++)
    {
      if(curoffset >= largestoffset) break;
      dary->add_value(grad_addr[curoffset]);
      curoffset++;
    }
    int mykey = paramid*2048+j;
    param_table_->update(mykey,v);
  }
}

template<>
void TypedTableDelegate<int, AdaGradValue>::Update(Param *param, int step){
  int paramid = param->id();
  int largestoffset = param->data().shape().Size();
  int splitsize = GlobalContext::Get()->num_table_servers()*split_size_;
  int splitoffset = largestoffset/splitsize;
  if (largestoffset%splitsize) splitoffset++;
  if (splitoffset > 1000000)
  {
    splitoffset = 1000000;
    splitsize = largestoffset/splitoffset + 1;
  }
  if (splitsize > 2048)VLOG(3)<<"Error:split size too much!!!";
  int curoffset = 0;

  const float * grad_addr = param->grad().dptr();
  for(int j = 0; j < splitsize; j++)
  {
    AdaGradValue v;
    v.set_version(step);
    DAryProto* dary=v.mutable_grad();
    dary->clear_value();
    for(int k = 0; k < splitoffset; k++)
    {
      if(curoffset >= largestoffset) break;
      dary->add_value(grad_addr[curoffset]);
      curoffset++;
    }
    int mykey = paramid*2048+j;
    param_table_->update(mykey,v);
  }
}

template<>
void TypedTableDelegate<int, SGDValue>::Put(Param * param){
  int paramid = param->id();
  int largestoffset = param->data().shape().Size();
  int splitsize = GlobalContext::Get()->num_table_servers()*split_size_;
  int splitoffset = largestoffset/splitsize;
  if (largestoffset%splitsize) splitoffset++;
  if (splitoffset > 1000000)
  {
    splitoffset = 1000000;
    splitsize = largestoffset/splitoffset + 1;
  }
  if (splitsize > 2048)VLOG(3)<<"Error:split size too much!!!";
  int curoffset = 0;
  const float * data_addr = param->data().dptr();
  for(int j = 0; j < splitsize; j++)
  {
    SGDValue v(example_);
    // sgd related hyper-parameters
    v.set_learning_rate_multiplier(param->learning_rate_multiplier());
    v.set_weight_decay_multiplier(param->weight_decay_multiplier());
    v.set_version(0);
    v.set_n_update(0);
    DAryProto* dary=v.mutable_data();
    DAryProto* grad=v.mutable_grad();
    dary->clear_value();
    for(int k = 0; k < splitoffset; k++)
    {
      if(curoffset >= largestoffset) break;
      dary->add_value(data_addr[curoffset]);
      grad->add_value(0.0f);
      curoffset++;
    }
    int mykey = paramid*2048+j;
    param_table_->put(mykey,v);
  }
}

template<>
void TypedTableDelegate<int, AdaGradValue>::Put(Param * param){
  int paramid = param->id();
  int largestoffset = param->data().shape().Size();
  int splitsize = GlobalContext::Get()->num_table_servers()*split_size_;
  int splitoffset = largestoffset/splitsize;
  if (largestoffset%splitsize) splitoffset++;
  if (splitoffset > 1000000)
  {
    splitoffset = 1000000;
    splitsize = largestoffset/splitoffset + 1;
  }
  if (splitsize > 2048)VLOG(3)<<"Error:split size too much!!!";
  int curoffset = 0;
  const float * data_addr = param->data().dptr();
  for(int j = 0; j < splitsize; j++)
  {
    AdaGradValue v(example_);
    v.set_version(0);
    v.set_n_update(0);
    DAryProto* dary=v.mutable_data();
    DAryProto* grad=v.mutable_grad();
    dary->clear_value();
    for(int k = 0; k < splitoffset; k++)
    {
      if(curoffset >= largestoffset) break;
      dary->add_value(data_addr[curoffset]);
      grad->add_value(0.0f);
      curoffset++;
    }
    int mykey = paramid*2048+j;
    param_table_->put(mykey,v);
  }
}
template<>
void TypedTableDelegate<VKey, AdaGradValue>::AsyncCollect(Param * param, int step){
  int paramid = param->id();
  int len = param->data().size();
  int splitsize = std::max(param->split_threshold(), len/GlobalContext::Get()->num_table_servers());
  int splitoffset = largestoffset/splitsize;
  if (largestoffset%splitsize) splitoffset++;
  if(splitsize<16777216)
    LOG(WARNING)<<"split of size "<<splitsize
    <<"  exceeds the size of max google protobuf message"
    <<" param length is "<<len <<" reset the splitsize to smaller one";
    splitoffset = 1000000;
    splitsize = largestoffset/splitoffset + 1;
  }
  if (splitsize > 2048)VLOG(3)<<"Error:split size too much!!!";
  int curoffset = 0;
  float * data_addr = param->mutable_data()->dptr();
  VKey key;
  key.set_version(step);
  for(int j = 0; j < splitsize; j++)
  {
    key.set_key(paramid*2048+j)
    AdaGradValue v = param_table_->async_get_collect(key, v);
    const DAryProto& data=v.data();
    for(int k = 0; k < splitoffset; k++)
    {
      if(curoffset >= largestoffset) break;
      //to pass new float to the params
      data_addr[curoffset] = data.value(k);
      curoffset++;
    }
  }
}


template<>
void TypedTableDelegate<int, AdaGradValue>::Get(Param * param, int step){
  int paramid = param->id();
  int largestoffset = param->data().shape().Size();
  int splitsize = GlobalContext::Get()->num_table_servers()*split_size_;
  int splitoffset = largestoffset/splitsize;
  if (largestoffset%splitsize) splitoffset++;
  if (splitoffset > 1000000)
  {
    splitoffset = 1000000;
    splitsize = largestoffset/splitoffset + 1;
  }
  if (splitsize > 2048)VLOG(3)<<"Error:split size too much!!!";
  int curoffset = 0;
  float * data_addr = param->mutable_data()->dptr();
  for(int j = 0; j < splitsize; j++)
  {
    int mykey = paramid*2048+j;
    AdaGradValue v = param_table_->get(mykey);
    const DAryProto& data=v.data();
    for(int k = 0; k < splitoffset; k++)
    {
      if(curoffset >= largestoffset) break;
      //to pass new float to the params
      data_addr[curoffset] = data.value(k);
      curoffset++;
    }
  }
}


template<>
void TypedTableDelegate<int, SGDValue>::Get(Param * param, int step){
  int paramid = param->id();
  int largestoffset = param->data().shape().Size();
  int splitsize = GlobalContext::Get()->num_table_servers()*split_size_;
  int splitoffset = largestoffset/splitsize;
  if (largestoffset%splitsize) splitoffset++;
  if (splitoffset > 1000000)
  {
    splitoffset = 1000000;
    splitsize = largestoffset/splitoffset + 1;
  }
  if (splitsize > 2048)VLOG(3)<<"Error:split size too much!!!";
  int curoffset = 0;
  float * data_addr = param->mutable_data()->dptr();
  for(int j = 0; j < splitsize; j++)
  {
    int mykey = paramid*2048+j;
    SGDValue v = param_table_->get(mykey);
    const DAryProto& data=v.data();
    for(int k = 0; k < splitoffset; k++)
    {
      if(curoffset >= largestoffset) break;
      //to pass new float to the params
      data_addr[curoffset] = data.value(k);
      curoffset++;
    }
  }
}



}  // namespace lapis

