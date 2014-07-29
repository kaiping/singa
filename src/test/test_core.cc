//  Copyright © 2014 Anh Dinh. All Rights Reserved.
//  main class for testing distributed memory layer
//
//  the command to run this should be:
//		mpirun -hostfile <host> -bycore -nooversubscribe
//				-n <num_servers> test --sync_update --system.conf=""


#include "core/common.h"
#include "core/table-registry.h"
#include "core/global-table.h"
#include "core/table.h"
#include "core/distributed-memory.h"
#include "core/memory-server.h"
#include "utils/global_context.h"
#include "glog/logging.h"
#include "gflags/gflags.h"

using namespace lapis;

DEFINE_bool(sync_update, false, "Synchronous put/update queue");
DEFINE_int32(num_server, 1, "default number of server");
DEFINE_string(system_conf, "system.conf", "configuration file for node roles");
DEFINE_string(model_conf, "model.conf", "DL model configuration file");

typedef TypedGlobalTable<int,int> Table;

MemoryServer* memory_server;

Table* init(int argc, char **argv){
	Table* t = CreateTable(0, 2, new Sharding::Mod, new Accumulators<int>::Sum,
            new Marshal<int>, new Marshal<int>);

	google::InitGoogleLogging(argv[0]);
	gflags::ParseCommandLineFlags(&argc, &argv, true);
	FLAGS_logtostderr = true;
	GlobalContext::Get()->Init(FLAGS_system_conf, FLAGS_model_conf);

	 NetworkThread::Init();
	    if (IsDistributedMemoryManager()) {
	      DistributedMemoryManager::Init();
	      DistributedMemoryManager::Get()->StartMemoryManager();
	      DistributedMemoryManager::Get()->AssignTables();
	    } else {
	      memory_server = new MemoryServer();
	      memory_server-> StartMemoryServer();
	    }
	return t;
}

void finish(){
	if (IsDistributedMemoryManager())
		DistributedMemoryManager::Get()->ShutdownServers();
	else
		memory_server->ShutdownMemoryServer();
}

void put(Table* table){
	LOG(INFO) << StringPrintf("Process %d: put...", NetworkThread::Get()->id());
	    //put, update then get
	for (int i = 0; i < 10; i++)
	   table->put(i, i);
	LOG(INFO) << StringPrintf("Process %d: Done put", NetworkThread::Get()->id());
}

void get(Table* table){
	LOG(INFO) << StringPrintf("Process %d: get...", NetworkThread::Get()->id());
	    for (int i = 0; i < 10; i++)
	      std::cout << "(" << i << ", " << table->get(i) << ")" << std::endl;
	LOG(INFO) << StringPrintf("Process %d: Done get", NetworkThread::Get()->id());
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
    get(table);
    update(table);
    get(table);
  } else { // worker, sleep while the network thread is processing put/get
    Sleep(7);
  }
  finish();
}


