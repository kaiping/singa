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
  virtual void Update(Param *param, int step)=0;
  virtual void Get(Param * param, int step)=0;
  virtual void Put(Param * param)=0;
  virtual void AsyncCollect(Param * param, int step)=0;
  virtual void AsyncGet(Param * param, int step)=0;
  virtual void SplitParams(const std::vector<Param *> &params, int wid)=0;

  void Update(const std::vector<Param *> &params, int step);
  void Get(const std::vector<Param *> &params, int step);
  void AsyncGet(const std::vector<Param *> &params, int step);
  void Put(const std::vector<Param *> &params);
  std::map<int, GlobalTable*> tables()=0;
};

template <typename K, typename V>
class TypedTableDelegate:public TableDelegate {
 public:
  explicit TypedTableDelegate(const SolverProto& proto);
  virtual void Update(Param *param, int step);
  virtual void Get(Param * param , int step);
  virtual void Put(Param * param);
  virtual std::map<int, GlobalTable*> tables(){
    return std::map<int, GlobalTable*> {0, param_table};
  }
  virtual void SplitParams(const std::vector<Param *> &params, int wid);
  void CreateTable(const SolverProto& solver);
  void set_example(const V& example){ example_=example; }
  virtual ~TypedTableDelegate();
 private:
  TypedGlobalTable<K, V>* CreateParamTable( const int id, int num_shards,
      UpdateHandler<V> *update, Sharder<K> *skey,
      Marshal<K> *mkey, Marshal<V> *mval) ;
 private:
  V example_;
  TypedGlobalTable<K,V> * param_table_;
  std::map<int, V*> splits_;
};

TableDelegate* CreateTableDelegate(const SolverProto& proto){
/**
 * need to know the tuple type to create parameter table
 */
  if(proto.method()==lapis::SolverProto::kSGD){
     delegate_=new lapis::TypedTableDelegate<VKey, lapis::SGDValue>(proto);
  }
  else{
    delegate_= new lapis::TypedTableDelegate<VKey, lapis::AdaGradValue>(proto);
  }
  return delegate;
}


template<typename K, typename V>
TypedTableDelegate<K,V>::TypedTableDelegate(const SolverProto& proto){
  VLOG(3)<<"In model controller";
  CreateTable(proto);
  if(proto.method()==lapis::SolverProto::kSGD)
    example_=proto.sgd();
  else
    exmaple_=proto.adagrad();
}

template<typename K, typename V>
void TypedTableDelegate<K,V>::CreateTable(const SolverProto& solver){
  auto update_handler=new UpdateHandler<V>(solver);
  param_table_= CreateParamTable(0, GlobalContext::Get()->num_table_servers(),
      update_handler,new Sharding::Mod, new Marshal<K>, new Marshal<V>);
}

template<class K, class V>
TypedGlobalTable<K, V>* TypedTableDelegate<K,V>::CreateParamTable(
    const int id, int num_shards, UpdateHandler<V>* update, Sharder<K> *skey,
    Marshal<K> *mkey, Marshal<V> *mval) {
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

template<class K, class V>
void TypedTableDelegate<K, V>::AsyncGet(Param * param, int step){
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
  VKey key;
  key.set_version(step);
  for(int j = 0; j < splitsize; j++)
  {
    key.set_key(paramid*2048+j);
    param_table_->async_get(key);
  }
}


template<>
void TypedTableDelegate<VKey ,SGDValue>::Update(Param *param,int step);
template<>
void TypedTableDelegate<VKey, AdaGradValue>::Update(Param *param, int step);
template<>
void TypedTableDelegate<VKey, SGDValue>::Put(Param * param);
template<>
void TypedTableDelegate<VKey ,AdaGradValue>::Put(Param * param);
template<>
void TypedTableDelegate<VKey ,SGDValue>::Get(Param * param, int step);
template<>
void TypedTableDelegate<VKey, AdaGradValue>::Get(Param * param, int step);
template<>
void TypedTableDelegate<VKey, AdaGradValue>::AsyncCollect(Param * param, int step);
template<>
void TypedTableDelegate<VKey, SGDValue>::AsyncCollect(Param * param, int step);





}  // namespace lapis
#endif  // INCLUDE_CORE_TABLE_DELEGATE_H_

