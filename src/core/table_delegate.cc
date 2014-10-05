// Copyright © 2014 Jinyang Gao. All Rights Reserved.
// 2014-07-18 16:33

#include <vector>

#include "core/table_delegate.h"
#include "core/disk-table.h"
#include "core/table.h"
#include "proto/model.pb.h"


namespace lapis {

void TableDelegate::Insert(const int id, int record_id, const Record& record){
  dynamic_cast<TDiskTable*>(tables_[id])->put(record_id, record);
}
void TableDelegate::Flush(const int id){
  dynamic_cast<TDiskTable*>(tables_[id])->finish_put();
}
void TableDelegate::Next(const int id, int *record_id, Record* record){
  int k;
  TDiskTable* table=dynamic_cast<TDiskTable*>(tables_[id]);
  if(!table->has_loaded())
    table->Load();
  if(table->done())
    table->Load();
   table->get(&k, record);
   table->Next();
}
void TableDelegate::Update(const std::vector<Param*> &params) {
  for(auto* param: params)
    Update(param);
  return;
}

void TableDelegate::Put(const std::vector<Param*> &params) {
  VLOG(3)<<"model controller put";
  if(GlobalContext::Get()->standalone())return;
  for(auto* param: params)
    Put(param);
}

void TableDelegate::Get(const std::vector<Param*> &params)
{
  if(GlobalContext::Get()->standalone())return;
  for(auto* param : params)
    Get(param);
  return;
}

template<>
void TypedTableDelegate<int ,SGDValue>::Update(Param *param){
  int paramid = param->id();
  int largestoffset = param->length();
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

    SGDValue v(example_);
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
void TypedTableDelegate<int, AdaGradValue>::Update(Param *param){
  int paramid = param->id();
  int largestoffset = param->length();
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
    AdaGradValue v(example_);
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
  int largestoffset = param->length();
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
    v.set_factor(param->factor());
    DAryProto* dary=v.mutable_data();
    dary->clear_value();
    for(int k = 0; k < splitoffset; k++)
    {
      if(curoffset >= largestoffset) break;
      dary->add_value(data_addr[curoffset]);
      curoffset++;
    }
    int mykey = paramid*2048+j;
    param_table_->put(mykey,v);
  }
}
template<>
void TypedTableDelegate<int ,AdaGradValue>::Put(Param * param){
  int paramid = param->id();
  int largestoffset = param->length();
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
    DAryProto* dary=v.mutable_data();
    dary->clear_value();
    for(int k = 0; k < splitoffset; k++)
    {
      if(curoffset >= largestoffset) break;
      dary->add_value(data_addr[curoffset]);
      curoffset++;
    }
    int mykey = paramid*2048+j;
    param_table_->put(mykey,v);
  }
}
template<>
void TypedTableDelegate<int ,SGDValue>::Get(Param * param){
  int paramid = param->id();
  int largestoffset = param->length();
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
  float * data_addr = param->mutable_data()->mutable_dptr();
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
template<>
void TypedTableDelegate<int, AdaGradValue>::Get(Param * param){
  int paramid = param->id();
  int largestoffset = param->length();
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
  float * data_addr = param->mutable_data()->mutable_dptr();
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
bool GetHandler<SGDValue>::Get(const SGDValue& from, SGDValue* to) {
  return true;
}
template<>
bool GetHandler<AdaGradValue>::Get(const AdaGradValue& from, AdaGradValue* to) {
  return true;
}
/*
template<>
UpdateHandler<AdaGradValue>::UpdateHandler(const SolverProto& solver){
}
}
template<>
bool UpdateHandler<AdaGradValue>::Update(AdaGradValue* data, const AdaGradValue& update){
  Vector dst(data->grad());
  Vector src(update->grad());
  Add(*dst, dst, src);
  return true;
}
*/

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
/*
template<>
void UpdateHandler<SGDValue>::UpdateHyperParams(const int step) {
  learning_rate_ = UpdateHyperParam(step, learning_rate_change_,
      learning_rate_change_steps_,
      base_learning_rate_,
      learning_rate_x_);
  momentum_ = UpdateHyperParam(step, sgd_proto_.momentum_change(),
      sgd_proto_.momentum_change_steps(),
      sgd_proto_.base_momentum(),
      sgd_proto_.momentum_x());
  weight_decay_ = UpdateHyperParam(step, sgd_proto_.weight_decay_change(),
      sgd_proto_.weight_decay_change_steps(),
      sgd_proto_.base_weight_decay(),
      sgd_proto_.weight_decay_x());
}
  */
}  // namespace lapis

