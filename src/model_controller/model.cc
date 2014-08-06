// Copyright © 2014 Jinyang Gao. All Rights Reserved.
// 2014-07-18 16:33

#include <vector>

#include "model_controller/myacc.h"
#include "model_controller/model.h"


namespace lapis {

ModelProto *ModelController::Init()
{
  VLOG(3)<<"In model controller";
  GlobalContext* gc=GlobalContext::Get();
  my_split_tpye_ = 0;
  my_machine_num_ = gc->num_memory_servers();
  my_split_size_ = 2;
  //start the lower level network part
  issinglemachine_ = gc->single();
  //start the lower level network part
  VLOG(3)<<"init network thread";
  NetworkThread::Init();
  VLOG(3)<<"Get network thread";
  net_ = NetworkThread::Get();
  if(!issinglemachine_){
    VLOG(3)<<"before create table num of machines "<<my_machine_num_;
    distributed_store_ = CreateTable(0, my_machine_num_, new Sharding::Mod,
        new MyAcc, new Marshal<int>, new Marshal<float_vector_message>);
    VLOG(3)<<"create table";
  }
  if(!issinglemachine_)
  {
    isdmm_ = IsDistributedMemoryManager();
    if (isdmm_) {
      dmm_ = DistributedMemoryManager::Get();
      DistributedMemoryManager::Init();
      VLOG(3)<<"finish init dmm"<<dmm_;
      dmm_->StartMemoryManager();
      VLOG(3)<<"finish start mem manager";
      dmm_->AssignTables();
      VLOG(3)<<"finish assign  mem manager";
    } else {
      ms_ = new MemoryServer();
      ms_-> StartMemoryServer();
      VLOG(3)<<"start mem server";
    }
  }
  my_rank_ = net_->id();
  int start_rank = gc->StartRankOf(lapis::kCoordinator);
  int end_rank = gc->EndRankOf(lapis::kCoordinator);
  iscoordinator_ = (my_rank_ <= end_rank && my_rank_ >= start_rank);
  ModelProto *model_proto=new ModelProto();
  if (iscoordinator_) {
    //do nothing?
  } else {
    net_->Read(start_rank,MTYPE_MC_CONFIG, dynamic_cast<Message*>(model_proto));
    VLOG(3)<<"work read model_proto";
  }

  return model_proto;
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
        float_vector_message mymessage;
        mymessage.clear_myfloat();
        for(int k = 0; k < splitoffset; k++)
        {
          if(curoffset >= largestoffset) break;
          mymessage.add_myfloat(grad_addr[curoffset]);
          curoffset++;
        }
        int mykey = paramid*2048+j;
        distributed_store_->update(mykey,mymessage);
      }
    }
  }
  return;
}

void ModelController::Put(const std::vector<Param*> &params)
{
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
      float_vector_message mymessage;
      mymessage.clear_myfloat();
      for(int k = 0; k < splitoffset; k++)
      {
        if(curoffset >= largestoffset) break;
        mymessage.add_myfloat(content_addr[curoffset]);
        curoffset++;
      }
      int mykey = paramid*2048+j;
      distributed_store_->put(mykey,mymessage);
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
      float_vector_message mymessage = distributed_store_->get(mykey);
      VLOG(3)<<"msg size "<<mymessage.myfloat_size();
      VLOG(3)<<splitoffset;
      for(int k = 0; k < splitoffset; k++)
      {
        if(curoffset >= largestoffset) break;
        //to pass new float to the params
        content_addr[curoffset] = mymessage.myfloat(k);
        curoffset++;
      }
    }
  }
  return;
}


void ModelController::CommenceBroadcast(const Message &modelconfig)
{
  if (iscoordinator_) {
    VLOG(3)<<"before broadcast";
    net_->Broadcast(MTYPE_MC_CONFIG,modelconfig);
    VLOG(3)<<"after Broadcast";
  }
}

void ModelController::CommenceSpecialConfig(const Message &modelconfig, int dst)
{
  if (iscoordinator_)
    net_->Send(dst,MTYPE_MC_CONFIG,modelconfig);
}

void ModelController::Finish() {
  if(issinglemachine_)return;
  isdmm_ ? dmm_->ShutdownServers() : ms_->ShutdownMemoryServer();
  return;
}

}  // namespace lapis
