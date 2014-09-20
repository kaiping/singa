// Copyright © 2014 Jinyang Gao. All Rights Reserved.
// 2014-07-18 16:33

#include <vector>

#include "model_controller/myacc.h"
#include "model_controller/model.h"
#include "core/disk-table.h"
#include "core/table.h"

namespace lapis {

ModelController::ModelController()
{
  VLOG(3)<<"In model controller";
  num_data_store_=0;
  num_param_store_=0;
  num_tables_=0;
  split_tpye_ = 0;
  split_size_ = 2;
  param_table_=nullptr;
  //start the lower level network part
  //start the lower level network part
}

void ModelController::PutData(int sid, int rid, const FloatVector &data){
  CHECK(disk_tables_.find(sid)!=disk_tables_.end());
  dynamic_cast<TDiskTable *>(disk_tables_[sid])->put(rid, data);
}

void ModelController::FlushData(int sid){
  CHECK(disk_tables_.find(sid)!=disk_tables_.end());
  dynamic_cast<TDiskTable*>(disk_tables_[sid])->finish_put();
}

void ModelController::GetData(int sid, Blob *blob) {
  CHECK(disk_tables_.find(sid)!=disk_tables_.end());
  TDiskTable* table= dynamic_cast<TDiskTable*>(disk_tables_.at(sid));
  if(!table->has_loaded()){
    table->Load();
  }
  int len=blob->record_length();
  FloatVector v;

  for(int i=0;i<blob->num();i++){
    int k;
    table->get(&k, &v);
    memcpy(blob->dptr+i*len, v.data().data(), len*sizeof(float));

    if(table->done()){
    	if (i<(blob->num()-1)){
             table->Load();
    	}
    }
    else
    	table->Next();

  }
}

void ModelController::Update(const std::vector<Param*> &params)
{
  if(GlobalContext::Get()->standalone())
  {
    for(auto* param: params)
    {
      const float * grad_addr = param->history().dptr;
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
    {
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

      const float * grad_addr = param->history().dptr;
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
	VLOG(2) << "update key ... " << mykey << " at process " << NetworkThread::Get()->id(); 
        param_table_->update(mykey,mymessage);
      }
    }
  }
  return;
}

void ModelController::Put(const std::vector<Param*> &params)
{
  if(GlobalContext::Get()->standalone())return;
  for(auto* param: params)
  {
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
	VLOG(2) << "put key ... " << mykey << " at process " << NetworkThread::Get()->id(); 
      param_table_->put(mykey,mymessage);
    }
  }
}

void ModelController::Get(const std::vector<Param*> &params)
{
  if(GlobalContext::Get()->standalone())return;
  for(auto* param : params)
  {
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
    float * content_addr = param->mutable_content().dptr;
    for(int j = 0; j < splitsize; j++)
    {
      int mykey = paramid*2048+j;
	VLOG(2) << "get key ... " << mykey << " at process " << NetworkThread::Get()->id(); 
      FloatVector mymessage = param_table_->get(mykey);
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

const std::map<int,GlobalTable*> ModelController::GetTables() {
  VLOG(3)<<"get tables";
  std::map<int,GlobalTable*> tables;
  VLOG(2)<<"disk table size "<<disk_tables_.size();
  for(auto& entry: disk_tables_){
    tables[entry.second->id()]=entry.second;
  }
  if(param_table_!=nullptr)
    tables[param_table_->id()]=dynamic_cast<GlobalTable*>(param_table_);
  VLOG(3)<<"finish get tables";
  return tables;
}


const std::map<int,int> ModelController::GetDataStoreTable() {
  std::map<int, int> store_table_map;
  for(auto& entry: disk_tables_) {
    store_table_map[entry.first]=entry.second->id();
  }
  return store_table_map;
}
const std::map<int,int> ModelController::GetParamStoreTable() {
  std::map<int, int> store_table_map;
  store_table_map[kParamStore]=param_table_->id();
  return store_table_map;
}

int ModelController::CreateDataStore(std::string name, int fixed_server_id) {
  VLOG(2)<<"create store for "<<name;
  int sid=2*num_data_store_+kDataStore;
  if(fixed_server_id>=0) {
    disk_tables_[sid]=CreateDiskTable(num_tables_, fixed_server_id, 256*10, name,
        new Marshal<int>, new Marshal<FloatVector>);
  }
  else{
    disk_tables_[sid]=CreateDiskTable(num_tables_, 256*10, name,
        new Marshal<int>, new Marshal<FloatVector>);
  }
  num_tables_++;
  num_data_store_++;
  return sid;
}

int ModelController::CreateParamStore() {
  if(GlobalContext::Get()->standalone()) return -1;
  param_table_= CreateTable(num_tables_, GlobalContext::Get()->num_table_servers(),
      new Sharding::Mod, new MyAcc, new Marshal<int>, new Marshal<FloatVector>);
  num_tables_++;
  return kParamStore;
}

void ModelController::CreateTables(const std::map<int, int>& tables){
  for(auto& entry: tables) {
    if(entry.first%2==kParamStore)
			param_table_ = CreateTable(entry.second,
					GlobalContext::Get()->num_table_servers(),
					new Sharding::Mod, new MyAcc, new Marshal<int>,
					new Marshal<FloatVector>);
    else
			disk_tables_[entry.first] = CreateDiskTable(entry.second, 256 * 10,
					StringPrintf("table_%d", entry.second), new Marshal<int>,
					new Marshal<FloatVector>);
  }
}

template<class K, class V>
TypedGlobalTable<K, V> *ModelController::CreateTable(
    int id, int num_shards, Sharder<K> *skey,
    Accumulator<V> *accum, Marshal<K> *mkey, Marshal<V> *mval) {
  TableDescriptor *info = new TableDescriptor(id, num_shards);
  info->key_marshal = mkey;
  info->value_marshal = mval;
  info->sharder = skey;
  info->accum = accum;
  info->partition_factory = new typename SparseTable<K, V>::Factory;
  auto table=new TypedGlobalTable<K, V>();
  table->Init(info);
  return table;
}

template<class K, class V>
TypedDiskTable<K,V>* ModelController::CreateDiskTable(int id, int max_size,
		string name, Marshal<K>* mkey, Marshal<V>* mval){
	DiskTableDescriptor *info = new DiskTableDescriptor(id, name, max_size);
	info->key_marshal = mkey;
	info->value_marshal = mval;
	TypedDiskTable<K,V> *t = new TypedDiskTable<K,V>(info);
	return t;
}

//  one desginated server stores the data
template<class K, class V>
TypedDiskTable<K,V>* ModelController::CreateDiskTable(int id,
    int fixed_server_id, int max_size,
		string name, Marshal<K>* mkey, Marshal<V>* mval){
	TypedDiskTable<K,V>* t = CreateDiskTable(id, max_size, name, mkey, mval);
	t->disk_info()->fixed_server_id = fixed_server_id;
	return t;
}

}  // namespace lapis
