// Copyright © 2014 Jinyang Gao. All Rights Reserved.
// 2014-07-18 16:33

#include <vector>

#include "model_controller/myacc.h"
#include "model_controller/model.h"


namespace lapis {

void ModelController::Init()
{
  VLOG(3)<<"In model controller";
  auto gc=GlobalContext::Get();
  my_split_tpye_ = 0;
  my_machine_num_ = gc->num_table_servers();
  my_split_size_ = 2;
  //start the lower level network part
  issinglemachine_ = gc->standalone();
  //start the lower level network part
}

void ModelController::PutData(int sid, int rid, const FloatVector &data){

}

void ModelController::GetData(int sid, Blob *blob) {

}

const std::map<int,int> ModelController::GetStoreTableMap() {
  std::map<int, int> store_table_map;
  for(auto& entry: tables_) {
    store_table_map[entry.first]=entry.second->id();
  }
  return store_table_map;
}

int ModelController::CreateDataStore() {
  int sid=2*num_data_store+kDataStore;
  tables_[sid]=CreateDiskTable(num_table_);
  num_table_++;
  num_data_store_++;
  return sid;
}

int ModelController::CreateParamStore() {
  if(issinglemachine_) return;
  int sid=kParamStore;
  VLOG(3)<<"before create table num of machines "<<my_machine_num_;
  tables_[sid]= CreateTable(num_table_, my_machine_num_, new Sharding::Mod,
        new MyAcc, new Marshal<int>, new Marshal<FloatVector>);
  param_table_=tables_[sid];
  num_table_++;
  VLOG(3)<<"create table";
  return kParamStore;
}

void ModelController::Update(const std::vector<Param*> &params)
{
  if(issinglemachine_)
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
  }

  if(!issinglemachine_)
  {
    for(auto* param: params)
    {
      int paramid = param->id();
      int largestoffset = param->length();
      int splitsize = my_machine_num_*my_split_size_;
      int splitoffset = largestoffset/splitsize;
      if (largestoffset%splitsize) splitoffset++;
      if (splitoffset > 1000000)
      {
        splitoffset = 1000000;
        splitsize = largestoffset/splitoffset + 1;
      }
      if (splitsize > 2048)VLOG(3)<<"Error:split size too much!!!";
      int curoffset = 0;

      const float * grad_addr = param->gradient().dptr;
      for(int j = 0; j < splitsize; j++)
      {
        FloatVector mymessage;
        mymessage.clear_data();
        for(int k = 0; k < splitoffset; k++)
        {
          if(curoffset >= largestoffset) break;
          mymessage.add_data(grad_addr[curoffset]);
          curoffset++;
        }
        int mykey = paramid*2048+j;
        param_table_->update(mykey,mymessage);
      }
    }
  }
  return;
}

void ModelController::Put(const std::vector<Param*> &params)
{
  VLOG(3)<<"model controller put";
  if(issinglemachine_)return;
  for(auto* param: params)
  {
    int paramid = param->id();
    int largestoffset = param->length();
    int splitsize = my_machine_num_*my_split_size_;
    int splitoffset = largestoffset/splitsize;
    if (largestoffset%splitsize) splitoffset++;
    if (splitoffset > 1000000)
    {
      splitoffset = 1000000;
      splitsize = largestoffset/splitoffset + 1;
    }
    if (splitsize > 2048)VLOG(3)<<"Error:split size too much!!!";
    int curoffset = 0;
    const float * content_addr = param->content().dptr;
    for(int j = 0; j < splitsize; j++)
    {
      FloatVector mymessage;
      mymessage.clear_data();
      for(int k = 0; k < splitoffset; k++)
      {
        if(curoffset >= largestoffset) break;
        mymessage.add_data(content_addr[curoffset]);
        curoffset++;
      }
      int mykey = paramid*2048+j;
      param_table_->put(mykey,mymessage);
    }
  }
}

void ModelController::Get(const std::vector<Param*> &params)
{
  if(issinglemachine_)return;
  for(auto* param : params)
  {
    int paramid = param->id();
    int largestoffset = param->length();
    int splitsize = my_machine_num_*my_split_size_;
    int splitoffset = largestoffset/splitsize;
    if (largestoffset%splitsize) splitoffset++;
    if (splitoffset > 1000000)
    {
      splitoffset = 1000000;
      splitsize = largestoffset/splitoffset + 1;
    }
    if (splitsize > 2048)VLOG(3)<<"Error:split size too much!!!";
    int curoffset = 0;
    float * content_addr = param->mutable_content().dptr;
    for(int j = 0; j < splitsize; j++)
    {
      int mykey = paramid*2048+j;
      FloatVector mymessage = param_table_->get(mykey);
      VLOG(3)<<"msg size "<<mymessage.data_size();
      VLOG(3)<<splitoffset;
      for(int k = 0; k < splitoffset; k++)
      {
        if(curoffset >= largestoffset) break;
        //to pass new float to the params
        content_addr[curoffset] = mymessage.data(k);
        curoffset++;
      }
    }
  }
  return;
}

void ModelController::CreateTables(std::map<int, int> tables){
  for(auto& entry: tables) {
    if(entry.first%2==kParamStore)
      tables_[entry.first]=CreateTable(entry.second);
    else
      tables_[entry.first]=CreateDiskTable(entry.second);
  }
}

template<class K, class V>
std::shared_ptr<TypedGlobalTable<K, V>> ModelController::CreateTable(
    int id, int num_shards, Sharder<K> *skey,
    Accumulator<V> *accum, Marshal<K> *mkey, Marshal<V> *mval) {
  TableDescriptor *info = new TableDescriptor(id, num_shards);
  info->key_marshal = mkey;
  info->value_marshal = mval;
  info->sharder = skey;
  info->accum = accum;
  info->partition_factory = new typename SparseTable<K, V>::Factory;
  auto table=make_shared<TypedGlobalTable<K, V>>(new TypedGlobalTable<K, V>());
  table->Init(info);
  return table;
}

}  // namespace lapis
