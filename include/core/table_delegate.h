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
/*
template<>
bool GetHandler<SGDValue>::Get(const SGDValue& from, SGDValue* to){
  return true;
}
template<>
bool GetHandler<AdaGradValue>::Get(const AdaGradValue& from, AdaGradValue* to){
  return true;
}
*/
template<typename V>
class UpdateHandler: public BaseUpdateHandler<V>{
 public:
  explicit UpdateHandler(const SolverProto& solver);
  virtual bool Update(V* data, const V& update);
};

template<>
class UpdateHandler<AdaGradValue>{
 public:
  explicit UpdateHandler(const SolverProto& solver);
  virtual bool Update(AdaGradValue* data, const AdaGradValue& update);
};

template<>
class UpdateHandler<SGDValue>{
 public:
  explicit UpdateHandler(const SolverProto& solver);
  virtual bool Update(SGDValue* data, const SGDValue& update);
  void UpdateHyperParams(const int step);
 private:
  int step_;
  float learning_rate_,  base_learning_rate_, gamma_;
  int learning_rate_change_steps_;
  float momentum_, weight_decay_;
  SGDValue::ChangeProto learning_rate_change_;
};


/***************************************************************************
 * Table Delegate
 **************************************************************************/
class TableDelegate {
 public:
  virtual ~TableDelegate();
  /**
   * create disk tables for train, val, test;
   * create parameter table
   */
  virtual void CreateTables(const SolverProto& solver);
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
template<>
void TypedTableDelegate<int ,SGDValue>::Update(Param *param);
template<>
void TypedTableDelegate<int, AdaGradValue>::Update(Param *param);
template<>
void TypedTableDelegate<int, SGDValue>::Put(Param * param);
template<>
void TypedTableDelegate<int ,AdaGradValue>::Put(Param * param);
template<>
void TypedTableDelegate<int ,SGDValue>::Get(Param * param);
template<>
void TypedTableDelegate<int, AdaGradValue>::Get(Param * param);

}  // namespace lapis
#endif  // INCLUDE_CORE_TABLE_DELEGATE_H_

