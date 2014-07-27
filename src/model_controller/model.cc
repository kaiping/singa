// Copyright © 2014 Jinyang Gao. All Rights Reserved.
// 2014-07-18 16:33


#include <vector>

#include "model_controller/myacc.h"
#include "model_controller/model.h"
namespace lapis {

Message ModelController::Init()
{
  GlobalContext* gc=GlobalContext::Get();
	
	my_split_tpye_ = 0;
	my_machine_num_ = gc->num_memory_servers();
	my_split_size_ = 2;

    //start the lower level network part
    FLAGS_logtostderr = true;
    FLAGS_logbuflevel = -1;

    net_ = NetworkThread::Get();
    my_rank_ = net_->id();

    isdmm_ = IsDistributedMemoryManager();
    NetworkThread::Init();

	if (isdmm_)
	{
    dmm_ = DistributedMemoryManager::Get();
    DistributedMemoryManager::Init();
    dmm_->StartMemoryManager();
    dmm_->AssignTables();
    }
    else{
    ms_ = new MemoryServer();
    ms_-> StartMemoryServer();
    }
    int start_rank=gc->StartRankOf(lapis::kCoordinator);
    int end_rank=gc->EndRankOf(lapis::kCoordinator);
    iscoordinator_ = (my_rank_ <= end_rank && my_rank_ >= start_rank);
    Message modelconfig;
	if (iscoordinator_)
    {
        //do nothing?
    }
    else
    {
        net_->Read(start_rank,MTYPE_MC_BROADCAST, &modelconfig);
    }
    distributed_store_ = CreateTable(0, my_machine_num_, new Sharding::Mod,
                                    new MyAcc, new Marshal<int>, new Marshal<float_vector_message>);
	return modelconfig;
}


void ModelController::Update(const std::vector<Param*> &params)
{
	for(auto* param: params)
    {
        int paramid = param->id();
        int splitoffset = param->length()/(my_machine_num_*my_split_size_);
        if (param->length()%(my_machine_num_*my_split_size_)) splitoffset++;
        int curoffset = 0;
        int largestoffset = param->length();
        const float * grad_addr = param->gradient();
        for(int j = 0; j < my_machine_num_*my_split_size_; j++)
        {
            float_vector_message mymessage;
            mymessage.clear_myfloat();
            for(int k = 0; k < splitoffset; k++)
            {
                if(curoffset >= largestoffset) break;
                mymessage.add_myfloat(grad_addr[curoffset]);
                curoffset++;
            }
            int mykey = paramid*my_machine_num_*my_split_size_+j;
            distributed_store_->update(mykey,mymessage);
        }
    }
    return;
}

void ModelController::Put(const std::vector<Param*> &params)
{
    for(auto* param: params)
    {
        int paramid = param->id();
        int splitoffset = param->length()/(my_machine_num_*my_split_size_);
        if (param->length()%(my_machine_num_*my_split_size_)) splitoffset++;
        int curoffset = 0;
        int largestoffset = param->length();
        const float * content_addr = param->content();
        for(int j = 0; j < my_machine_num_*my_split_size_; j++)
        {
            float_vector_message mymessage;
            mymessage.clear_myfloat();
            for(int k = 0; k < splitoffset; k++)
            {
                if(curoffset >= largestoffset) break;
                mymessage.add_myfloat(content_addr[curoffset]);
                curoffset++;
            }
            int mykey = paramid*my_machine_num_*my_split_size_+j;
            distributed_store_->put(mykey,mymessage);
        }
    }
    return;
}

void ModelController::Get(const std::vector<Param*> &params)
{
    for(auto* param : params)
    {
        int paramid = param->id();
        int splitoffset = param->length()/(my_machine_num_*my_split_size_);
        if (param->length()%(my_machine_num_*my_split_size_)) splitoffset++;
        int curoffset = 0;
        int largestoffset = param->length();
        float * content_addr = param->mutable_content();
        for(int j = 0; j < my_machine_num_*my_split_size_; j++)
        {
            int mykey = paramid*my_machine_num_*my_split_size_+j;
            float_vector_message mymessage = distributed_store_->get(mykey);
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


void ModelController::CommenceBroadcast(const Message &modelconfig) {
  if (iscoordinator_)
    net_->Broadcast(MTYPE_MC_BROADCAST,modelconfig);
}
void ModelController::Finish() {
  isdmm_ ? dmm_->ShutdownServers() : ms_->ShutdownMemoryServer();
}

}  // namespace lapis
