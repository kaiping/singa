// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-10-03 10:35

#ifndef INCLUDE_CORE_TABLE_DELEGATE_H_
#define INCLUDE_CORE_TABLE_DELEGATE_H_
#include <stdlib.h>
#include <vector>
#include <glog/logging.h>
#include <google/protobuf/message.h>
#include <string>

#include "net/param.h"
#include "utils/global_context.h"
#include "utils/common.h"
#include "core/common.h"
#include "core/global-table.h"
#include "core/disk-table.h"
#include "core/sparse-table.h"
#include "core/table.h"
#include "core/table_server.h"
#include "proto/model.pb.h"


namespace lapis {
using std::string;
using google::protobuf::Message;
using TDiskTable=TypedDiskTable<int, Record>;

template<typename V>
class GetHandler{
 public:
 /**
  * return true for Success; false for failure;
  */
  bool Get(const V& from, V* to);
};

template<typename V>
class UpdateHandler: public BaseUpdateHandler<V>{
 public:
  explicit UpdateHandler(const SolverProto& solver);
  virtual bool Update(V* data, const V& update);
};
template<>
class UpdateHandler<SGDValue>{
 public:
   explicit UpdateHandler(const SolverProto& solver){
     step_=0;
     base_learning_rate_=solver.sgd().base_learning_rate();
     learning_rate_x_=solver.sgd().learning_rate_x();
     learning_rate_change_=solver.sgd().learning_rate_change();
     learning_rate_change_steps_=solver.sgd().learning_rate_change_steps();
     momentum_=solver.sgd().momentum();
     weight_decay_=solver.sgd().weight_decay();
   }

  virtual bool Update(SGDValue* data, const SGDValue& update){
    return true;
  }
  void UpdateHyperParams(const int step) {}
 private:
  int step_;
  float learning_rate_,  base_learning_rate_, learning_rate_x_;
  int learning_rate_change_steps_;
  float momentum_, weight_decay_;
  SGDValue::ChangeProto learning_rate_change_;
};

class TableDelegate {
 public:

  /**
   * create disk tables for train, val, test;
   * create parameter table
   */
  virtual void CreateTables(const SolverProto& solver)=0;
  virtual void Update(Param *param)=0;
  virtual void Get(Param * param)=0;
  virtual void Put(Param * param)=0;

  void Update(const std::vector<Param *> &params);
  void Get(const std::vector<Param *> &params);
  void Put(const std::vector<Param *> &params);

  void Insert(const int id, int record_id, const Record& record);
  void Next(const int id, int *record_id, Record* record);
  void Flush(const int id);

  TypedDiskTable<int,Record>* CreateDiskTable(const int id, int max_size, string name, Marshal<int>* mkey, Marshal<Record>* mval);
  TypedDiskTable<int,Record>* CreateDiskTable(const int id, int fixed_server_id, int max_size, string name, Marshal<int>* mkey, Marshal<Record>* mval);

 const std::map<int, GlobalTable*>& tables(){return tables_;}
 protected:
  std::map<int, GlobalTable*> tables_;
};

template <typename K, typename V>
class TypedTableDelegate:public TableDelegate {
 public:
  TypedTableDelegate();
  virtual void CreateTables(const SolverProto& solver);
  virtual void Update(Param *param);
  virtual void Get(Param * param);
  virtual void Put(Param * param);

  void set_example(const V& example){ example_=example; }
 private:
  TypedGlobalTable<K, V>* CreateParamTable( const int id, int num_shards, UpdateHandler<V> *update, Sharder<K> *skey,  Marshal<K> *mkey, Marshal<V> *mval) ;
 private:
  V example_;
  int split_tpye_,split_size_;
  TypedGlobalTable<K,V> * param_table_;
};

template<typename K, typename V>
TypedTableDelegate<K,V>::TypedTableDelegate(){
  VLOG(3)<<"In model controller";
  split_tpye_ = 0;
  split_size_ = 2;
}

template<typename K, typename V>
void TypedTableDelegate<K,V>::CreateTables(const SolverProto& solver){
  auto update_handler=new UpdateHandler<V>(solver);
  param_table_= CreateParamTable(0, GlobalContext::Get()->num_table_servers(),
      update_handler,new Sharding::Mod, new Marshal<K>, new Marshal<V>);
  tables_[kTrain]=CreateDiskTable(static_cast<const int>(kTrain), 256*10, std::to_string(kTrain), new Marshal<int>, new Marshal<Record>);
  tables_[kVal]=CreateDiskTable(static_cast<const int>(kVal), 256*10, std::to_string(kTrain), new Marshal<int>, new Marshal<Record>);
  tables_[kTest]=CreateDiskTable(static_cast<const int>(kTest), 256*10, std::to_string(kTrain), new Marshal<int>, new Marshal<Record>);
  tables_[0]=param_table_;
}

template<class K, class V>
TypedGlobalTable<K, V>* TypedTableDelegate<K,V>::CreateParamTable( const int id, int num_shards, UpdateHandler<V>* update, Sharder<K> *skey,  Marshal<K> *mkey, Marshal<V> *mval) {
  TableDescriptor *info = new TableDescriptor(id, num_shards);
  info->key_marshal = mkey;
  info->value_marshal = mval;
  info->sharder = skey;
  // TODO update accum
  info->accum = update;
  info->partition_factory = new typename SparseTable<K, V>::Factory;
  auto table=new TypedGlobalTable<K, V>();
  table->Init(info);
  VLOG(3)<<"after create param table ";
  VLOG(3)<<"table shards num "<<table->num_shards();
  return table;
}
TypedDiskTable<int,Record>* TableDelegate::CreateDiskTable(const int id, int max_size, string name, Marshal<int>* mkey, Marshal<Record>* mval){
  DiskTableDescriptor *info = new DiskTableDescriptor(id, name, max_size);
  info->key_marshal = mkey;
  info->value_marshal = mval;
  TypedDiskTable<int,Record> *t = new TypedDiskTable<int,Record>(info);
  return t;
}
//  one desginated server stores the data
TypedDiskTable<int,Record>* TableDelegate::CreateDiskTable(const int id, int fixed_server_id, int max_size, string name, Marshal<int>* mkey, Marshal<Record>* mval){
	TDiskTable* t = CreateDiskTable(id, max_size, name, mkey, mval);
	t->disk_info()->fixed_server_id = fixed_server_id;
  VLOG(3)<<"after create disk table "<<name;
  VLOG(3)<<"table shards num "<<t->num_shards();
	return t;
}
}  // namespace lapis
#endif  // INCLUDE_CORE_TABLE_DELEGATE_H_

