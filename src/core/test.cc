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

int main(int argc, char** argv){


	TypedGlobalTable<int, int>* test_table =
		CreateTable(0, 2, new Sharding::Mod, new Accumulators<int>::Sum, new Marshal<int>, new Marshal<int>);

	InitServers(argc, argv);


	if (IsDistributedMemoryManager()){
		DistributedMemoryManager::Get()->AssignTables();

		LOG(INFO) << StringPrintf("Process %d: put:", NetworkThread::Get()->id());
		//put, update then get
		for (int i=0; i<100; i++)
			test_table->put(i,i);

		LOG(INFO) << StringPrintf("Process %d: Done put ...", NetworkThread::Get()->id());
		
		LOG(INFO) << StringPrintf("Process %d: get:", NetworkThread::Get()->id());
		for (int i=0; i<100; i++)
			std::cout << "("<< i << ", "<< test_table->get(i)<< ")" << std::endl;
		LOG(INFO) << StringPrintf("Process %d: Done get ...", NetworkThread::Get()->id());

/*
		LOG(INFO) << StringPrintf("Process %d: update:", NetworkThread::Get()->id());
		for (int i=0; i<10; i++)
			test_table->update(i, 2*i);
		LOG(INFO) << StringPrintf("Process %d: Done update ...", NetworkThread::Get()->id());


		LOG(INFO) << StringPrintf("Process %d: get:", NetworkThread::Get()->id());
		for (int i=0; i<10; i++)
				std::cout << "("<< i << ", "<< test_table->get(i)<< ")" << std::endl;
		LOG(INFO) << StringPrintf("Process %d: Done get ...", NetworkThread::Get()->id());
*/

	}
	else{ // worker, sleep while the network thread is processing put/get
		Sleep(15);
	}

	Finish();
}


