// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-10-03 10:35

#ifndef INCLUDE_CORE_TABLE_DELEGATE_H_
#define INCLUDE_CORE_TABLE_DELEGATE_H_
#include <stdlib.h>
#include <glog/logging.h>
#include <string>
#include <vector>
#include <map>

#include "darray/dary.h"
#include "net/param.h"
#include "utils/global_context.h"
#include "utils/common.h"
#include "core/global-table.h"
#include "core/common.h"
#include "core/sparse-table.h"
#include "core/table.h"
#include "proto/model.pb.h"


namespace lapis {
using std::string;

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
  virtual ~TableDelegate(){};
  /**
   * create disk tables for train, val, test;
   * create parameter table
   */
  virtual void Update(Param *param, int step)=0;
  virtual void Get(Param * param, int step)=0;
  virtual void Put(Param * param)=0;

  virtual void AsyncGet(Param * param, int step)=0;
  virtual void AsyncCollect(Param * param, int step)=0;

  virtual void SplitParams(const std::vector<Param *> &params, int wid)=0;

  virtual const std::map<int, GlobalTable*> tables()=0;

  void Update(const std::vector<Param *> &params, int step);
  void Get(const std::vector<Param *> &params, int step);
  void Put(const std::vector<Param *> &params);
  void AsyncGet(const std::vector<Param *> &params, int step);
};

template <typename K, typename V>
class TypedTableDelegate:public TableDelegate {
 public:
  explicit TypedTableDelegate(const SolverProto& proto);

  virtual void Update(Param *param, int step);
  virtual void Get(Param * param , int step);
  virtual void Put(Param * param);
  virtual void AsyncGet(Param * param, int step);
  virtual void AsyncCollect(Param * param, int step);

  virtual void SplitParams(const std::vector<Param *> &params, int wid);
  virtual const std::map<int, GlobalTable*> tables(){
    std::map<int, GlobalTable*> ret;
    ret[0]=param_table_;
    return  ret;
  }

  void set_example(const V& example){ example_=example; }
  virtual ~TypedTableDelegate();
 private:
  TypedGlobalTable<K, V>* CreateParamTable( const int id, int num_shards,
      UpdateHandler<V> *update, Sharder<K> *skey,
      Marshal<K> *mkey, Marshal<V> *mval) ;
 private:
  int kMaxSplits_;
  V example_;
  TypedGlobalTable<K,V> * param_table_;
  std::map<int, vector<std::pair<int, int>>> split_map_;
};

template<class K, class V>
TypedTableDelegate<K,V>::~TypedTableDelegate(){
  delete param_table_;
}

struct VKeySharder :public Sharder<VKey> {
  int operator() (const VKey& k, int shards) {
    return k.key()%shards;
  }
};

inline bool operator==(const VKey& k1, const VKey& k2) {
  return k1.key()==k2.key();
}

template<>
TypedTableDelegate<VKey,SGDValue>::TypedTableDelegate(const SolverProto& proto);
template<>
TypedTableDelegate<VKey,AdaGradValue>::TypedTableDelegate(const SolverProto& proto);


TableDelegate* CreateTableDelegate(const SolverProto& proto);
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
void TypedTableDelegate<K, V>::SplitParams(const vector<Param *>& params, int wid) {
  int total_splits=0;
  int group_size=GlobalContext::Get()->group_size();
  for(auto param: params){
    const DAry& dary=param->data();
    int id=param->id()*group_size+wid;
    int local_size=dary.allocated();
    int splitsize=param->split_threshold();
    /*
    //int splitsize=std::max(param->split_threshold(), local_size/num_servers);
    if(splitsize==local_size/num_servers&&local_size%num_servers!=0)
    splitsize+=1;
    */
    if(splitsize>=16777216){
      LOG(WARNING)<<"split of size "<<splitsize
        <<"  exceeds the size of max google protobuf message, i.e., 64MB"
        <<" param length is "<<local_size <<", reset the split threshold to 1000000";
      splitsize = 1000000;
    }
    int nsplits=local_size/splitsize+local_size%splitsize;
    vector<std::pair<int, int>> splits;
    for(auto j = 0, pos=0; j < nsplits; j++) {
      int len=pos+splitsize<local_size?splitsize:local_size-pos;
      splits.push_back(std::make_pair(id*kMaxSplits_+j,len));
      pos+=len;
    }
    split_map_[param->id()]=splits;
    total_splits+=splits.size();
  }
  CHECK(total_splits<kMaxSplits_)
    <<"total splits exceeds kMaxSplits, raise kMaxSplits in solver config";
}

template<class K, class V>
void TypedTableDelegate<K, V>::Update(Param *param, int step){
  int offset = 0;
  const float * dptr = param->grad().dptr();
  K key;
  key.set_version(step);
  for(auto& entry: split_map_[param->id()]) {
    V v(example_);
    // sgd related hyper-parameters
    v.set_version(step);
    DAryProto* grad=v.mutable_grad();
    grad->clear_value();
    for(int k = 0; k < entry.second; k++){
      grad->add_value(dptr[offset]);
      offset++;
    }
    key.set_key(entry.first);
    param_table_->update(key, v);
  }
}

template<class K, class V>
void TypedTableDelegate<K, V>::Get(Param * param, int step){
  float* dptr=param->mutable_data()->dptr();
  K key;
  key.set_version(step);
  int offset=0;
  for(auto entry: split_map_[param->id()]) {
    key.set_key(entry.first);
    V v=param_table_->get(key);
    for(auto x: v.data().value()){
      dptr[offset++]=x;
    }
    CHECK_EQ(v.data().value_size(), entry.second);
  }
  CHECK_EQ(offset, param->data().allocated());
}

template<class K, class V>
void TypedTableDelegate<K, V>::AsyncGet(Param * param, int step){
  auto splits=split_map_[param->id()];
  K key;
  key.set_version(step);
  V v;
  for(auto entry: splits) {
    key.set_key(entry.first);
    param_table_->async_get(key, &v);
  }
}

template<class K, class V>
void TypedTableDelegate<K, V>::AsyncCollect(Param * param, int step){
  float * dptr = param->mutable_data()->dptr();
  auto& splits=split_map_[param->id()];
  std::map<int, int> offset;
  int pos=0;
  for(auto entry: splits){
    offset[entry.first]=pos;
    pos+=entry.second;
  }
  K key;
  key.set_version(step);
  int nget=0;
  while(nget<splits.size()){
    for(auto i=0;i<splits.size();++i){
      V v;
      key.set_key(splits[i].first);
      if(param_table_->async_get_collect(&key,&v)){
        int k=offset[key.key()];
        for(auto x: v.data().value())
          dptr[k++]=x;
        nget++;
      } else {
        sleep(0.001);
      }
    }
  }
}


template<>
void TypedTableDelegate<VKey, SGDValue>::Put(Param * param);
template<>
void TypedTableDelegate<VKey ,AdaGradValue>::Put(Param * param);

}  // namespace lapis
#endif  // INCLUDE_CORE_TABLE_DELEGATE_H_

