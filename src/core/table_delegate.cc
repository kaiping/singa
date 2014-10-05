// Copyright © 2014 Jinyang Gao. All Rights Reserved.
// 2014-07-18 16:33

#include <vector>

#include "core/table_delegate.h"
#include "core/disk-table.h"
#include "core/table.h"
#include "proto/model.pb.h"


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
  learning_rate_x_=solver.sgd().learning_rate_x();
  learning_rate_change_=solver.sgd().learning_rate_change();
  learning_rate_change_steps_=solver.sgd().learning_rate_change_steps();
  momentum_=solver.sgd().momentum();
  weight_decay_=solver.sgd().weight_decay();
}
bool UpdateHandler<SGDValue>::Update(SGDValue* data, const SGDValue& update){
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
}


void TableDelegate::CreateTables(const SolverProto& solver) {
  tables_[kTrain]=CreateDiskTable(static_cast<const int>(kTrain), 256*10, std::to_string(kTrain), new Marshal<int>, new Marshal<Record>);
  tables_[kVal]=CreateDiskTable(static_cast<const int>(kVal), 256*10, std::to_string(kTrain), new Marshal<int>, new Marshal<Record>);
  tables_[kTest]=CreateDiskTable(static_cast<const int>(kTest), 256*10, std::to_string(kTrain), new Marshal<int>, new Marshal<Record>);
}
TDiskTable* TableDelegate::CreateDiskTable(const int id, int max_size, string name, Marshal<int>* mkey, Marshal<Record>* mval){
  DiskTableDescriptor *info = new DiskTableDescriptor(id, name, max_size);
  info->key_marshal = mkey;
  info->value_marshal = mval;
  TypedDiskTable<int,Record> *t = new TypedDiskTable<int,Record>(info);
  return t;
}
//  one desginated server stores the data
TDiskTable* TableDelegate::CreateDiskTable(const int id, int fixed_server_id, int max_size, string name, Marshal<int>* mkey, Marshal<Record>* mval){
	TDiskTable* t = CreateDiskTable(id, max_size, name, mkey, mval);
	t->disk_info()->fixed_server_id = fixed_server_id;
  VLOG(3)<<"after create disk table "<<name;
  VLOG(3)<<"table shards num "<<t->num_shards();
	return t;
}

TableDelegate::~TableDelegate() {
  for(auto& entry: tables_)
    delete entry.second;
}
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


}  // namespace lapis

