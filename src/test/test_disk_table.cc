//  Copyright © 2014 Anh Dinh. All Rights Reserved.
//  main class for testing distributed memory layer
//
//  the command to run this should be:
//		mpirun -hostfile <host> -bycore -nooversubscribe
//				-n <num_servers> test -sync_update


#include "core/global-table.h"
#include "core/common.h"
#include "core/disk-table.h"
#include "core/table.h"
#include "core/table_server.h"
#include "utils/global_context.h"
#include <gflags/gflags.h>
#include "proto/model.pb.h"
#include "worker.h"

DEFINE_int32(record_size,100, "# elements per float vector");
DEFINE_int32(block_size, 10, "# records per block, multiple blocks per table");
DEFINE_int32(table_size, 500, "# records per table");
DEFINE_string(system_conf, "examples/imagenet12/system.conf", "configuration file for node roles");
DEFINE_string(model_conf, "examples/imagenet12/model.conf", "DL model configuration file");
using namespace lapis;

typedef map<int, GlobalTable*> Map;
Map tables;

//  put random message to the pointers
void create_random_message(FloatVector* message){
	for (int i=0; i<FLAGS_record_size; i++)
		message->add_data(rand());
}

void create_disk_table(int id){
	DiskTableDescriptor *info = new DiskTableDescriptor(id, "disk_test",
			FLAGS_block_size);
	info->key_marshal = new Marshal<int>();
	info->value_marshal = new Marshal<FloatVector>();
	tables[id] = new TypedDiskTable<int,FloatVector>(info);
}

void run_coordinator(shared_ptr<NetworkThread> network, int tid){
	VLOG(3) << "coordinator running";
	// wait for wokers to be up
	RegisterWorkerRequest req;
	for (int i=0; i<network->size()-1; i++)
		network->Read(MPI::ANY_SOURCE, MTYPE_REGISTER_WORKER, &req);

	VLOG(3) << "put data";
	// put data in
	TypedDiskTable<int, FloatVector>* table = static_cast<TypedDiskTable<int,
			FloatVector>*>(tables[tid]);
	for (int i=0; i<FLAGS_table_size; i++){
		FloatVector message;
		create_random_message(&message);
		table->put(i, message);
	}
	table->finish_put();


	for (int i=0; i<network->size()-1; i++){
		EmptyMessage end_msg;
		network->Read(i,MTYPE_WORKER_END, &end_msg);
	}

	EmptyMessage shutdown_msg;
	for (int i = 0; i < network->size() - 1; i++) {
		network->Send(i, MTYPE_WORKER_SHUTDOWN, shutdown_msg);
	}
	network->Flush();
	network->Shutdown();
}

void run_worker(shared_ptr<NetworkThread> network){
	VLOG(3) << "worker running";
	TableServer* ts = new TableServer();
	ts->StartTableServer(tables);

  network->Flush();
  network->Send(GlobalContext::kCoordinatorRank, MTYPE_WORKER_END, EmptyMessage());
  EmptyMessage msg;

  int src = 0;
  network->Read(GlobalContext::kCoordinatorRank, MTYPE_WORKER_SHUTDOWN, &msg, &src);
  network->Flush();
  network->Shutdown();
}

int main(int argc, char **argv) {
	create_disk_table(0);

	FLAGS_logtostderr = 1;
	google::InitGoogleLogging(argv[0]);
	gflags::ParseCommandLineFlags(&argc, &argv, true);


	// Note you can register you own layer/edge/datasource here
	//
	// Init GlobalContext
	auto gc = lapis::GlobalContext::Get(FLAGS_system_conf, FLAGS_model_conf);
	//start network thread
			shared_ptr<NetworkThread> network = NetworkThread::Get();

	if (network->id() == network->size()-1)
		run_coordinator(network,0);
	else
		run_worker(network);

	return 0;
}


