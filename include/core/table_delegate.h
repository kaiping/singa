// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-10-03 10:35

#ifndef INCLUDE_CORE_TABLE_DELEGATE_H_
#define INCLUDE_CORE_TABLE_DELEGATE_H_

#include <vector>
#include <google/protobuf/message.h>

#include "net/param.h"
#include "utils/global_context.h"
#include "core/common.h"
#include "core/global-table.h"
#include "core/disk-table.h"
#include "core/sparse-table.h"
#include "core/table.h"
#include "core/table_server.h"
#include "proto/model.pb.h"


namespace lapis {

using google::protobuf::Message;
using TDiskTable=TypedDiskTable<int, Record>;
class TableDelegate {
 public:

  /**
   * create disk tables for train, val, test;
   * create parameter table
   */
  virtual const std::map<int, GlobalTable*>& CreateTables()=0;
  virtual void Update(Param *param)=0;
  virtual void Get(Param * param)=0;
  virtual void Put(Param * param)=0;


  void Update(const std::vector<Param *> &params);
  void Get(const std::vector<Param *> &params);
  void Put(const std::vector<Param *> &params);

  void Put(Phase phase, int record_id, const Record& record);
  void Get(Phase phase, int *record_id, Record* record);
  void Flush(int table_id);

  const std::map<int, GlobalTable*>& tables(){return tables_;}
 private:
  std::map<int, GlobalTable*> tables_;
};

template <typename K, typename V>
class TypedTableDelegate {
 public:
  virtual const std::map<int, GlobalTable*>& CreateTables()=0;
  virtual void Update(Param *param)=0;
  virtual void Get(Param * param)=0;
  virtual void Put(Param * param)=0;

 private:
  void set_example(const V& exmaple){ exmaple_=example; }
  TypedGlobalTable<K, V>* CreateParamTable( int id, int num_shards, Sharder<K> *skey,  Marshal<K> *mkey, Marshal<V> *mval) ;
  TypedDiskTable<K,V>* CreateDiskTable(int id, int max_size, string name, Marshal<K>* mkey, Marshal<V>* mval);
  TypedDiskTable<K,V>* CreateDiskTable(int id, int fixed_server_id, int max_size, string name, Marshal<K>* mkey, Marshal<V>* mval);

 private:
  V example_;
  int split_tpye_,split_size_;
  TypedGlobalTable<K,V> * param_table_;
};
template<class K, class V>
TypedGlobalTable<K, V>* TypedTableDelegate::CreateParamTable( int id, int num_shards, Sharder<K> *skey, Accumulator<V> *accum, Marshal<K> *mkey, Marshal<V> *mval) {
  TableDescriptor *info = new TableDescriptor(id, num_shards);
  info->key_marshal = mkey;
  info->value_marshal = mval;
  info->sharder = skey;
  info->accum = new UpdateHandler<V>;
  info->partition_factory = new typename SparseTable<K, V>::Factory;
  auto table=new TypedGlobalTable<K, V>();
  table->Init(info);
  VLOG(3)<<"after create param table ";
  VLOG(3)<<"table shards num "<<table->num_shards();
  return table;
}
template<class K, class V>
TypedDiskTable<K,V>* TypedTableDelegate::CreateDiskTable(int id, int max_size, string name, Marshal<K>* mkey, Marshal<V>* mval){
  DiskTableDescriptor *info = new DiskTableDescriptor(id, name, max_size);
  info->key_marshal = mkey;
  info->value_marshal = mval;
  TypedDiskTable<K,V> *t = new TypedDiskTable<K,V>(info);
  return t;
}
//  one desginated server stores the data
template<class K, class V>
TypedDiskTable<K,V>* TypedTableDelegate::CreateDiskTable(int id, int fixed_server_id, int max_size, string name, Marshal<K>* mkey, Marshal<V>* mval){
	TDiskTable* t = CreateDiskTable(id, max_size, name, mkey, mval);
	t->disk_info()->fixed_server_id = fixed_server_id;
  VLOG(3)<<"after create disk table "<<name;
  VLOG(3)<<"table shards num "<<t->num_shards();
	return t;
}
TableDelegate* CreateTableDelegate(const SolverProto::Method& method){
  if(method()==SolverProto::kSGD){
    return new TypedTableDelegate<int, SGDValue>();
  }
  else{
    return new TypedTableDelegate<int, AdaGradValue>();
  }
}

template<typename V>
class GetHandler{
 public:
 /**
  * return true for Success; false for failure;
  */
  bool Get(const V& from, V* to);
};

template<typename V>
class UpdateHandler{
 public:
  explicit UpdateHandler(const SovlerProto& solver);
  bool Update(V* data, const V& update);

 private:
  int threshold_;
};
template<>
class UpdateHandler<SGDValue>{
 public:
  explicit UpdateHandler(const SovlerProto& solver);
  bool Update(V* data, const V& update);

  void UpdateHyperParams(const int step);
 private:
  int threshold_;
  int step_;
  float learning_rate_,  base_learning_rate_, learning_rate_x_;
  Solver::Change learning_rate_change_;
  int learning_rate_change_steps_,
  float momentum_, weight_decay_,
};

}  // namespace lapis
#endif  // INCLUDE_CORE_TABLE_DELEGATE_H_

