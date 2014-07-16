//  Copyright © 2014 Anh Dinh. All Rights Reserved.
//  main class for testing distributed memory layer
//
//  the command to run this should be:
//		mpirun -hostfile <host> -bycore -nooversubscribe
//				-n <num_servers> [-nosyn] test

#include "core/common.h"

using namespace lapis;

DEFINE_bool(sync_update, false, "Synchronous put/update queue");

//  first, create table
//  then, launch the memory servers
int main(int argc*, char** argv){
	TypedGlobalContext<int, int> typed_g_context(new Sharding::Mod,
									new Accumulator<int>::Sum, new Marshall<int>, new Marshall<int>);

	TypedGlobalTable<int, int>* test_table = CreateTable(0, typed_g_context);

	InitServers(argc, argv);

	if (IsDistributedMemoryManager()){
		DistributedMemoryManager::Get()->AssignTables();

		//put, update then get
		for (int i=0; i<100; i++)
			test_table->put(i,i);

		for (int i=0; i<100; i++)
			test_table->update(i, 2*i);

		for (int i=0; i<100; i++)
			std::cout << "("<< i << ", "<< test_table->get(i)<< ")" << std::endl;

		// then send message to flush/stop other workers
		if (IsDistributedMemoryManager())
			DistributedMemoryManager::Get()->ShutdownServers();
	}

}


