// Copyright © 2014 Jinyang Gao. All Rights Reserved.
// 2014-07-18 16:33

#include <vector>

#include "core/table_delegate.h"
#include "core/disk-table.h"
#include "core/table.h"

namespace lapis {

TDiskTable* CreateDiskTable(int id, int max_size,
		string name, Marshal<int>* mkey, Marshal<Record>* mval){
	return t;
}
TableDelegate::TableDelegate()
{
  VLOG(3)<<"In model controller";
  split_tpye_ = 0;
  split_size_ = 2;
  param_table_=nullptr;
}

template<typename K, typename V>
std::map<int, GlobalTable*> TableDelegate::CreateTables(){
  tables[0]= CreateParamTable(0, GlobalContext::Get()->num_table_servers(),
      new Sharding::Mod, new Marshal<K>, new Marshal<V>);
  tables[kTrain]=CreateDiskTable(kTrain, 256*10, string(id), new Marshal<int>, new Marshal<Record>);
  tables[kVal]=CreateDiskTable(kVal, 256*10, string(id), new Marshal<int>, new Marshal<Record>);
  tables[kTest]=CreateDiskTable(kTest, 256*10, string(id), new Marshal<int>, new Marshal<Record>);
  param_table_=tables[0];
  return tables;
}

void TableDelegate::Put(Phase phase, int record_id, const Record& record){
  tables_[phase]->put(record_id, record);
}
void TableDelegate::Flush(int table_id){
  tables_[phase]->Flush();
}
void TableDelegate::Get(Phase phase, int *record_id, Record* record){
  int k;
  TDiskTable* table=tables_[phase];
  if(!table->has_loaded())
    table->Load();
  if(table->done())
    table->Load();
  table->get(&k, record);
}
void TableDelegate::Update(const std::vector<Param*> &params) {
  if(GlobalContext::Get()->standalone())
  {
    for(auto* param: params)
    {
      const float * grad_addr = param->gradient().dptr;
      float * content_addr = param->mutable_content().dptr;
      int largestoffset = param->length();
      for(int j = 0; j < largestoffset; j++)
      {
        content_addr[j] += grad_addr[j];
      }
    }
    return;
  }else {
    for(auto* param: params)
      Update<V>(param);
  }
  return;
}


template<>
void TypedTableDelegate<SGDValue>::Update(Param *param){
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

  const float * grad_addr = param->gradient().dptr();
  for(int j = 0; j < splitsize; j++)
  {

    SGDValue v(example_);
    DAryProto* dary=value.mutable_grad();
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
void TypedTableDelegate<AdaGradValue>::Update(Param *param){
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

  const float * grad_addr = param->gradient().dptr();
  for(int j = 0; j < splitsize; j++)
  {

    SGDValue v(example_);
    DAryProto* dary=value.mutable_grad();
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


void TableDelegate::Put(const std::vector<Param*> &params)
{
  VLOG(3)<<"model controller put";
  if(GlobalContext::Get()->standalone())return;
  for(auto* param: params)
    Put(param);
}
template<>
void TypedTableDelegate<SGDValue>::Put(Param * param){
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
    if(param->has_factor())
      v.set_factor(param->factor());
    DAryProto* dary=v.mutable_data();
    dary->clear_value();
    for(int k = 0; k < splitoffset; k++)
    {
      if(curoffset >= largestoffset) break;
      dary->add_value(content_addr[curoffset]);
      curoffset++;
    }
    int mykey = paramid*2048+j;
    param_table_->put(mykey,v);
  }
}
template<>
void TypedTableDelegate<AdaGradValue>::Put(Param * param){
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
      dary->add_value(content_addr[curoffset]);
      curoffset++;
    }
    int mykey = paramid*2048+j;
    param_table_->put(mykey,v);
  }
}
void TableDelegate::Get(const std::vector<Param*> &params)
{
  if(GlobalContext::Get()->standalone())return;
  for(auto* param : params)
    Get(param);
  return;
}
template<>
void TypedTableDelegate<SGDValue>::Get(Param * param){
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
  float * data_addr = param->mutable_data().dptr;
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
void TypedTableDelegate<AdaGradValue>::Get(Param * param){
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
  float * data_addr = param->mutable_data().dptr;
  for(int j = 0; j < splitsize; j++)
  {
    int mykey = paramid*2048+j;
    AdaGradV v = param_table_->get(mykey);
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
bool GetHandler<SGDValue>::Get(SGDValue* from, SGDValue* to) {

}
template<>
bool GetHandler<AdaGradValue>::Get(AdaGradValue* from, AdaGradValue* to) {

}

tmplate<>
bool UpdateHandler::<SGDValue>::Update(SGDValue* data, SGDValue* update){
}
tmplate<>
bool UpdateHandler::<AdaGradValue>::Update(AdaGradValue* data, AdaGradValue* update){
  /*
  Vector dst(data->grad());
  Vector src(update->grad());
  Add(*dst, dst, src);
  */
  return true;
}

float UpdateHyperParam(int step, Solver::ChangeProto change, int change_steps, float a, float b) {
  float ret = 0., r = 0.;
  switch (change) {
    case Solver::kFixed:
      ret = a;
      break;
    case Solver::kLinear:
      // a is init, b is the final
      r = step * 1.0  / change_steps;
      ret = (1.0 - r) * a + r * b;
      break;
    case Solver::kExponential:
      // a is init, b is the final, from convnet
      CHECK_EQ(a, 2 * b) << "final value should be the half";
      ret = a / pow(2, step * 1. / change_steps);
      break;
    case Solver::kInverse_t:
      // a is init, b is the final, from convnet
      CHECK_EQ(a, 2 * b) << "final value should be the half";
      ret = a / (1. + step * 1. / b);
      break;
    case Solver::kStep:
      // a is the base learning rate, b is gamma, from caffe
      ret = a * pow(b, step / change_steps);
      break;
    default:
      LOG(ERROR) << "Wrong hyper-parameter update method";
  }
  return ret;
}
template<>
void UpdateHandler<SGDValue>::UpdateHyperParams(const int step) {
  learning_rate_ = UpdateHyperParam(step, learning_rate_change_,
      learning_rate_change_steps_,
      base_learning_rate_,
      learning_rate_x_);
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

}  // namespace lapis

