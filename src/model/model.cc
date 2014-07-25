// Copyright © 2014 Jinyang Gao. All Rights Reserved.
// 2014-07-18 16:33

#include "mc/model_controller.h"

#include <vector>

#include "mc/myacc.h"
#include "core/common.h"
#include "core/table-registry.h"
#include "core/global-table.h"
#include "core/table.h"
#include "core/distributed-memory.h"
#include "core/memory-server.h"

namespace lapis {

void ModelController::Init(int machine_num,int split_tpye,int split_size)
{
	my_split_tpye_ = split_tpye;
	my_machine_num_ = machine_num;
	my_split_size_ = split_size;
	distributed_store_ = CreateTable(0, my_machine_num, new Sharding::Mod, new MyAcc, new Marshal<int>, new Marshal<int>);

	if (IsDistributedMemoryManager()){
    DistributedMemoryManager::Get()->AssignTables();
  }
	return;
}

void ModelController::Update(const std::vector<Param*> * params)
{
	for(int i = 0; i < params->size(); i++)
    {
        int paramid = (*params)[i]->id();
        int splitoffset = (*params)[i].length()/(my_machine_num_*my_split_size_);
        if ((*params)[i].length()%(my_machine_num_*my_split_size_)) splitoffset++;
        int curoffset = 0;
        int largestoffset = (*params)[i].length();
        const float * grad_addr = (*params)[i]->Gradient();
        for(int j = 0; j < my_machine_num_*my_split_size_; j++)
        {
            float_vector_message mymessage;
            mymessage.clear_myfloat();
            for(int k = 0; k < splitoffset; k++)
            {
                if(curoffset >= largestoffset)continue;
                mymessage.add_myfloat(grad_addr[curoffset]);
                curoffset++;
            }
            int mykey = paramid*my_machine_num_*my_split_size_+j;
            distributed_store_->update(mykey,mymessage);
        }
    }
    return;
}

void ModelController::Put(const std::vector<Param*> * params)
{
    for(int i = 0; i < params->size(); i++)
    {
        int paramid = (*params)[i]->id();
        int splitoffset = (*params)[i].length()/(my_machine_num_*my_split_size_);
        if ((*params)[i].length()%(my_machine_num_*my_split_size_)) splitoffset++;
        int curoffset = 0;
        int largestoffset = (*params)[i].length();
        const float * content_addr = (*params)[i]->Content();
        for(int j = 0; j < my_machine_num_*my_split_size_; j++)
        {
            float_vector_message mymessage;
            mymessage.clear_myfloat();
            for(int k = 0; k < splitoffset; k++)
            {
                if(curoffset >= largestoffset)continue;
                mymessage.add_myfloat(content_addr[curoffset]);
                curoffset++;
            }
            int mykey = paramid*my_machine_num_*my_split_size_+j;
            distributed_store_->put(mykey,mymessage);
        }
    }
    return;
}

void ModelController::GetParam(std::vector<Param*> * params)
{
    for(int i = 0; i < params->size(); i++)
    {
        int paramid = (*params)[i].id();
        int splitoffset = (*params)[i].length()/(my_machine_num_*my_split_size_);
        if ((*params)[i].length()%(my_machine_num_*my_split_size_)) splitoffset++;
        int curoffset = 0;
        int largestoffset = (*params)[i].length();
        float * content_addr = (*params)[i]->MutableContent();
        for(int j = 0; j < my_machine_num_*my_split_size_; j++)
        {
            int mykey = paramid*my_machine_num_*my_split_size_+j;
            float_vector_message mymessage = distributed_store_->get(mykey);
            for(int k = 0; k < splitoffset; k++)
            {
                if(curoffset >= params[i].length())continue;
                //to pass new float to the params
                content_addr[curoffset] = mymessage.myfloat(k);
                curoffset++;
            }
        }
    }
    return;
}


}  // namespace lapis
