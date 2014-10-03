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
std::map<int, GlobalTable*> ModelController::CreateTables(){
  std::map<int, TDiskTable*> tables;
  tables[0]= CreateTable(0, GlobalContext::Get()->num_table_servers(),
      new Sharding::Mod, new MyAcc, new Marshal<int>, new Marshal<TupleValue>);
  tables[kTrain]=CreateDiskTable(kTrain, 256*10, string(id), new Marshal<int>, new Marshal<Record>);
  tables[kVal]=CreateDiskTable(kVal, 256*10, string(id), new Marshal<int>, new Marshal<Record>);
  tables[kTest]=CreateDiskTable(kTest, 256*10, string(id), new Marshal<int>, new Marshal<Record>);
  param_table_=tables[0];
  return tables;
}
void ModelController::Update(const std::vector<Param*> &params)
{
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
  VLOG(3)<<"after create param table ";
  VLOG(3)<<"table shards num "<<table->num_shards();
  return table;
}

template<class K, class V>
TypedDiskTable<K,V>* ModelController::CreateDiskTable(int id, int max_size,
		string name, Marshal<K>* mkey, Marshal<V>* mval){
	DiskTableDescriptor *info = new DiskTableDescriptor(id, name, max_size);
	info->key_marshal = mkey;
	info->value_marshal = mval;
	TypedDiskTable<K,V> *t = new TypedDiskTable<K,V>(info);
  VLOG(3)<<"after create disk table "<<name<< " max size "<<t->disk_info()->max_size;
  VLOG(3)<<"table shards num "<<t->num_shards();
	return t;
}

//  one desginated server stores the data
template<class K, class V>
TypedDiskTable<K,V>* ModelController::CreateDiskTable(int id,
    int fixed_server_id, int max_size,
		string name, Marshal<K>* mkey, Marshal<V>* mval){
	TypedDiskTable<K,V>* t = CreateDiskTable(id, max_size, name, mkey, mval);
	t->disk_info()->fixed_server_id = fixed_server_id;
  VLOG(3)<<"after create disk table "<<name;
  VLOG(3)<<"table shards num "<<t->num_shards();
	return t;
}

}  // namespace lapis
