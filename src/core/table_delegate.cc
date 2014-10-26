// Copyright © 2014 Jinyang Gao. All Rights Reserved.
// 2014-07-18 16:33

#include <vector>

#include "core/table_delegate.h"
#include "core/disk-table.h"
#include "core/table.h"
#include "proto/model.pb.h"

#include "da/dary.h"

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
bool UpdateHandler<AdaGradValue>::Get(const VKey& key, const AdaGradValue &val, AdaGradValue* ret) {
  //LOG(INFO)<<"key version "<<key.version()<<" val version "<<val.version();
  if(key.version()<=val.version()){
    DAryProto* retdat=ret->mutable_data();
    retdat->clear_value();
    for(auto x: val.data().value())
      retdat->add_value(x);
    return true;
  }

  return false;
}

/*********************************************************************
 * SGDValue
 **********************************************************************/
bool UpdateHandler<SGDValue>::Get(const VKey& key, const SGDValue &val, SGDValue* ret) {
  //LOG(INFO)<<"key version "<<key.version()<<" val version "<<val.version();
  if(key.version()<=val.version()){
    DAryProto* retdat=ret->mutable_data();
    retdat->clear_value();
    for(auto x: val.data().value())
      retdat->add_value(x);
    return true;
  }
  return false;
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
  //LOG(INFO)<<"update for "<<data->id()<<" version "<<data->version()<<" "<<update.version();
  CHECK_EQ(data->version(), update.version())<<data->id()<<" "<<data->threshold()<<" "<<data->n_update();
  data->set_n_update(data->n_update()+1);
  if(data->n_update()==data->threshold()){
    UpdateHyperParams(data->version());
  }
  int len=data->data().value_size();
  float* history=data->mutable_grad()->mutable_value()->mutable_data();
  const float* grad=update.grad().value().data();
  CHECK_EQ(len, update.grad().value_size());
  CHECK_EQ(len, data->grad().value_size());
  float* dptr=data->mutable_data()->mutable_value()->mutable_data();
  float lr=learning_rate_*data->learning_rate_multiplier();
  float w=weight_decay_*data->weight_decay_multiplier();
  // hist=hist-lr*grad
  DAry::arymath().madd(history, lr, grad, history, len);
  // hist=hist-lr*weight*param
  if(w>0)
    DAry::arymath().madd(history, lr*w, dptr, history, len);

  if(data->n_update()==data->threshold()){
    // param+=history/n, /data->n_update()
    DAry::arymath().sub(dptr, dptr, history, len);
    float upp=0.f;
    for(int i=0;i<len;i++)
      upp+=fabs(history[i]);
    LOG(INFO)<<"update for "<<data->id()<<" "<<upp/len;
    // hist=hist*mom
    DAry::arymath().mul(history, momentum_, history, len);
    data->set_n_update(0);
    data->set_version(update.version()+1);
    //LOG(INFO)<<"update version for "<<data->id()<<" from "<<update.version();
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
      // notice it is step/change_steps, not step*1.0/change_steps
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

/********************************************************************
 * Table Delegate
 * *************************************************************/
void TableDelegate::HandleShardAssignment() {
	LOG(INFO) << "Handle Shard Assignment";
  ShardAssignmentRequest shard_req;
  auto mpi=NetworkThread::Get();
  mpi->Read(GlobalContext::kCoordinator, MTYPE_SHARD_ASSIGNMENT, &shard_req);
  //  request read from coordinator
  auto _tables=tables();
  for (int i = 0; i < shard_req.assign_size(); i++) {
    const ShardAssignment &a = shard_req.assign(i);
    GlobalTable *t = _tables.at(a.table());
    t->get_partition_info(a.shard())->owner = a.new_worker();
    //LOG(INFO) << StringPrintf("Process %d is assigned shard (%d,%d)", NetworkThread::Get()->id(), a.table(), a.shard());
  }
  EmptyMessage empty;
  mpi->Send(GlobalContext::kCoordinator, MTYPE_SHARD_ASSIGNMENT_DONE, empty);
  LOG(INFO)<<"Finish handle shard assignment";
}

TableDelegate* CreateTableDelegate(const SolverProto& proto){
/**
 * need to know the tuple type to create parameter table
 */
  TableDelegate *delegate;
  if(proto.method()==lapis::SolverProto::kSGD){
     delegate=new lapis::TypedTableDelegate<VKey, lapis::SGDValue>(proto);
  }
  else{
    delegate= new lapis::TypedTableDelegate<VKey, lapis::AdaGradValue>(proto);
  }
  NetworkThread::Get()->RegisterCallback(MTYPE_SHARD_ASSIGNMENT,
                         boost::bind(&TableDelegate::HandleShardAssignment, delegate));

  return delegate;
}


template<>
TypedTableDelegate<VKey,AdaGradValue>::TypedTableDelegate (const SolverProto& proto){
  auto* update_handler=new UpdateHandler<AdaGradValue>(proto);
  param_table_= CreateParamTable(0, GlobalContext::Get()->num_table_servers(),
      update_handler,new VKeySharder, new Marshal<VKey>, new Marshal<AdaGradValue>);

  example_=proto.adagrad();
  kMaxSplits_=proto.max_splits();
}


template<>
TypedTableDelegate<VKey,SGDValue>::TypedTableDelegate (const SolverProto& proto){
  auto* update_handler=new UpdateHandler<SGDValue>(proto);
  param_table_= CreateParamTable(0, GlobalContext::Get()->num_table_servers(),
      update_handler,new VKeySharder, new Marshal<VKey>, new Marshal<SGDValue>);

  example_=proto.sgd();
  kMaxSplits_=proto.max_splits();
}

void TableDelegate::Update(const std::vector<Param*> &params, int step) {
  for(auto* param: params)
    Update(param,step);
  return;
}

void TableDelegate::Put(const std::vector<Param*> &params) {
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
void TypedTableDelegate<VKey, SGDValue>::Put(Param * param){
  int offset = 0;
  int groupsize=GlobalContext::Get()->group_size();
  const float * data_addr = param->data().dptr();
  LOG(INFO)<<"param id "<<param->id()<<" name "<<param->name()
    <<" "<<param->partition()<<" "<<groupsize;
  for(auto& entry: param_splits_map_[param->id()]) {
    SGDValue v(example_);
    // sgd related hyper-parameters
    v.set_learning_rate_multiplier(param->learning_rate_multiplier());
    v.set_weight_decay_multiplier(param->weight_decay_multiplier());
    v.set_version(0);
    v.set_n_update(0);
    v.set_id(param->id());
    if(!param->partition())
      v.set_threshold(groupsize);
    else
      v.set_threshold(1);
    DAryProto* dary=v.mutable_data();
    DAryProto* grad=v.mutable_grad();
    dary->clear_value();
    grad->clear_value();
    for(int k = 0; k < entry.second; k++){
      dary->add_value(data_addr[offset]);
      grad->add_value(0.0f);
      offset++;
    }
    VKey key;
    key.set_version(0);
    key.set_key(entry.first);
    param_table_->put(key, v);
  }
}

template<>
void TypedTableDelegate<VKey, AdaGradValue>::Put(Param * param){
  int offset = 0;
  const float * data_addr = param->data().dptr();
  for(auto& entry: param_splits_map_[param->id()]) {
    AdaGradValue v(example_);
    // sgd related hyper-parameters
    v.set_version(0);
    v.set_n_update(0);
    DAryProto* dary=v.mutable_data();
    DAryProto* grad=v.mutable_grad();
    dary->clear_value();
    grad->clear_value();
    for(int k = 0; k < entry.second; k++){
      dary->add_value(data_addr[offset]);
      grad->add_value(0.0f);
      offset++;
    }
    VKey key;
    key.set_version(0);
    key.set_key(entry.first);
    param_table_->put(key, v);
  }
}


}  // namespace lapis

