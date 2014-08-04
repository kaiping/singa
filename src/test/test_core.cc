//  Copyright © 2014 Anh Dinh. All Rights Reserved.
//  main class for testing distributed memory layer
//
//  the command to run this should be:
//		mpirun -hostfile <host> -bycore -nooversubscribe
//				-n <num_servers> test -sync_update


#include "core/common.h"
#include "core/table-registry.h"
#include "core/global-table.h"
#include "core/table.h"
#include "core/distributed-memory.h"
#include "core/memory-server.h"

using namespace lapis;

DEFINE_bool(sync_update, false, "Synchronous put/update queue");
DEFINE_int32(num_server, 1, "default number of server");

int main(int argc, char** argv){

void get(Table* table){
	LOG(INFO) << StringPrintf("Process %d: get...", NetworkThread::Get()->id());
	    for (int i = 0; i < 10; i++)
	    	LOG(INFO) << StringPrintf("(%d,%d)", i, table->get(i));
	LOG(INFO) << StringPrintf("Process %d: Done get", NetworkThread::Get()->id());
}

void async_get(Table* table){
	LOG(INFO) << StringPrintf("Process %d: async get...", NetworkThread::Get()->id());
	int count = 0;
	    for (int i = 0; i < 10; i++){
	    	int v;
	    	if (table->async_get(i, &v))
	    		LOG(INFO) << StringPrintf("(%d,%d)", i, v);
	    	else{
	    		LOG(INFO) << StringPrintf("(%d,NULL)",i);
	    		count++;
	    	}
	    }
	    LOG(INFO) << "Collecting asynchronously ...";
	    while (count>0){
	    	int k,v;
	    	if (table->async_get_collect(&k,&v)){
	    		LOG(INFO) << StringPrintf("(%d,%d)", k, v);
	    		count--;
	    	}
	    	else
	    		sleep(1);
	    }


	LOG(INFO) << StringPrintf("Process %d: Done async get", NetworkThread::Get()->id());
}

void update(Table* table){
	 LOG(INFO) << StringPrintf("Process %d: update...", NetworkThread::Get()->id());
	    for (int i = 0; i < 10; i++)
	      table->update(i, 3);
	 LOG(INFO) << StringPrintf("Process %d: Done update", NetworkThread::Get()->id());
}

int main(int argc, char **argv) {
  Table *table = init(argc, argv);
  if (IsDistributedMemoryManager()) {
    put(table);
    async_get(table);
    update(table);
    get(table);
  } else { // worker, sleep while the network thread is processing put/get
    Sleep(7);
  }
  finish();
}


