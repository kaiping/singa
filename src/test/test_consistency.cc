//  Copyright © 2014 Anh Dinh. All Rights Reserved.

//  Testing the unbalance in spliting parameter vectors.

#include "core/global-table.h"
#include "core/common.h"
#include "core/disk-table.h"
#include "core/table.h"
#include "core/table_server.h"
#include "utils/global_context.h"
#include <gflags/gflags.h>
#include "proto/model.pb.h"
#include "proto/common.pb.h"
#include "worker.h"
#include "coordinator.h"
#include "utils/common.h"
#include "utils/proto_helper.h"

#include <cmath>
#include <stdlib.h>
#include <vector>
#include <iostream>
#include <fstream>

using namespace lapis;
using std::vector;

//DEFINE_bool(sync_update, false, "Synchronous put/update queue");
DEFINE_string(db_backend, "lmdb", "backend db");
DEFINE_string(system_conf, "examples/imagenet12/system.conf", "configuration file for node roles");
DEFINE_string(model_conf, "examples/imagenet12/model.conf", "DL model configuration file");
DEFINE_int64(threshold,1000000, "max # of parameters in a vector");
DEFINE_int32(iterations,5,"numer of get/put iterations");
DEFINE_int32(workers,2,"numer of workers doing get/put");
#ifndef FLAGS_v
  DEFINE_int32(v, 3, "vlog controller");
#endif
struct AnhUpdateHandler: BaseUpdateHandler<int,int>{
	bool Update(int *a, const int &b){
		*a = *a+b; //replace
		return true;
	}

  bool Get(const int k, const int &val, int *ret){
    if(val<k+2&&val!=k){
      VLOG(3)<<"get false";
      return false;
    }
    else{
      *ret = val;
      return true;
    }
  }
};

typedef map<int, GlobalTable*> Map;
Map tables;
shared_ptr<NetworkThread> network;
shared_ptr<GlobalContext> context;
std::vector<ServerState*> server_states;
TableServer *table_server;
TableDelegate *delegate;

int num_keys=5;

void create_mem_table(int id, int num_shards){

	TableDescriptor *info = new TableDescriptor(id, num_shards);
	  info->key_marshal = new Marshal<int>();
	  info->value_marshal = new Marshal<int>();
	  info->sharder = new Sharding::Mod;
	  info->accum = new AnhUpdateHandler;
	  info->partition_factory = new typename SparseTable<int, int>::Factory;
	  auto table=new TypedGlobalTable<int, int>();
	  table->Init(info);
	  tables[id] = table;
}

void coordinator_assign_tables(int id){
	for (int i = 0; i < context->num_procs() 	; ++i) {
	    RegisterWorkerRequest req;
	    int src = 0;
	    //  adding memory server.
	    if (context->IsTableServer(i)) {
	      network->Read(MPI::ANY_SOURCE, MTYPE_REGISTER_WORKER, &req, &src);
	      server_states.push_back(new ServerState(i));
	    }
	  }
	  LOG(INFO) << " All servers registered and started up. Ready to go";
	  //  set itself as the current worker for the table
	  tables[id]->worker_id_ = network->id();

	  // memory servers are specified in global context. Round-robin assignment

	    VLOG(3)<<"num of shards"<<tables[id]->num_shards()<<" for table"<< id;

	    int server_idx = 0;
	    for (int shard = 0; shard < tables[id]->num_shards(); ++shard) {
	      ServerState &server = *server_states[server_idx];
	      LOG(INFO) << "Assigning table ("<<id<<","<<shard<<") to server "
	                <<server_states[server_idx]->server_id;

	      // TODO(Anh) may overwrite this field if #shards>#table_servers
	      server.shard_id = shard;
	      server.local_shards.insert(new TaskId(id, shard));
	      server_idx = (server_idx + 1) % server_states.size();
	    }

	  VLOG(3)<<"table assignment";
	  //  then send table assignment
	  ShardAssignmentRequest req;
	  for (size_t i = 0; i < server_states.size(); ++i) {
	    ServerState &server = *server_states[i];
	    for (auto * task: server.local_shards) {
	      ShardAssignment *s  = req.add_assign();
	      s->set_new_worker(server.server_id);
	      s->set_table(task->table);
	      s->set_shard(task->shard);
	      //  update local tables
	      CHECK(tables.find(task->table)!=tables.end());
	      GlobalTable *t = tables.at(task->table);
	      t->get_partition_info(task->shard)->owner = server.server_id;
	      delete task;
	    }
	  }
	  VLOG(3)<<"finish table assignment, req size "<<req.assign_size();
	  network->SyncBroadcast(MTYPE_SHARD_ASSIGNMENT, MTYPE_SHARD_ASSIGNMENT_DONE, req);
	  VLOG(3)<<"finish table server init";
}


void worker_table_init(){
	table_server = new TableServer();
	table_server->StartTableServer(tables);
	VLOG(3) << "done starting table server";
}

double random_double(){
	return static_cast<double>(rand())/static_cast<double>(RAND_MAX);
}

// popular table with random large or small messages.
// the message distribution specified in FLAGS_large_precentage
void coordinator_load_data(){
	auto table = static_cast<TypedGlobalTable<int,int>*>(tables[0]);

	int nservers=context->num_table_servers();

	for (int i=1; i<num_keys; i++){

		table->put(i,i);
	}
	VLOG(3) << "Coordinator done loading ..., from process "<<NetworkThread::Get()->id();
}

void get(TypedGlobalTable<int,int>* table){
  double start , end;
  int v;
  for (int i=0; i<num_keys; i++)
    table->async_get(i, &v);


  int key=0;
  int val=0;

  LOG(INFO)<<"start collect key";
  for (int i=1; i<num_keys; i++){
    while(!table->async_get_collect(&key, &val))
      Sleep(0.001);
    LOG(INFO)<<"collect key "<<key<<" with val "<<val;
  }
}

void update(TypedGlobalTable<int,int>* table){
  if(NetworkThread::Get()->id()==0)
    sleep(10);
  LOG(INFO)<<"start update";
  for (int i=0; i<num_keys; i++)
    table->update(i,1);
}

void worker_test_data(){
  auto table = static_cast<TypedGlobalTable<int,int>*>(tables[0]);

  get(table);
  update(table);
  get(table);
}

void shutdown(){
	if (context->AmICoordinator()){
		EmptyMessage msg;
		for (int i=0; i<context->num_procs()-1; i++)
			network->Read(MPI::ANY_SOURCE, MTYPE_WORKER_END, &msg);
		 EmptyMessage shutdown_msg;
		  for (int i = 0; i < network->size() - 1; i++) {
		    network->Send(i, MTYPE_SHUTDOWN, shutdown_msg);
		  }
		  network->Flush();
		  network->Shutdown();
	}
	else{
	  network->Flush();

	  network->Send(context->num_procs()-1, MTYPE_WORKER_END, EmptyMessage());

	  EmptyMessage msg;

	  network->Read(context->num_procs()-1, MTYPE_SHUTDOWN, &msg);

	  if (context->AmITableServer())
		  table_server->ShutdownTableServer();

	  network->Shutdown();
	}
}

void HandleShardAssignment() {

  ShardAssignmentRequest shard_req;
  auto mpi=NetworkThread::Get();
  mpi->Read(GlobalContext::kCoordinator, MTYPE_SHARD_ASSIGNMENT, &shard_req);
  //  request read from coordinator
  for (int i = 0; i < shard_req.assign_size(); i++) {
    const ShardAssignment &a = shard_req.assign(i);
    GlobalTable *t = tables.at(a.table());
    t->get_partition_info(a.shard())->owner = a.new_worker();
    //LOG(INFO) << StringPrintf("Process %d is assigned shard (%d,%d)", NetworkThread::Get()->id(), a.table(), a.shard());
  }
  EmptyMessage empty;
  mpi->Send(GlobalContext::kCoordinator, MTYPE_SHARD_ASSIGNMENT_DONE, empty);
  VLOG(3)<<"finish handle shard assignment **";
}


int main(int argc, char **argv) {
	FLAGS_logtostderr = 1;
	int provided;
	MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
	google::InitGoogleLogging(argv[0]);
	gflags::ParseCommandLineFlags(&argc, &argv, true);

	context = GlobalContext::Get(FLAGS_system_conf);
	network = NetworkThread::Get();

	ModelProto model;
	ReadProtoFromTextFile(FLAGS_model_conf.c_str(), &model);

	create_mem_table(0,context->num_table_servers());

	if (context->AmICoordinator()){
		VLOG(3) << "Coordinator process rank = " << NetworkThread::Get()->id();
		coordinator_assign_tables(0);
		coordinator_load_data();

		network->barrier();
	}
	else{
		if (context->AmITableServer()){
			worker_table_init();
			HandleShardAssignment();
			network->barrier();
		}
		else{
			VLOG(3) << "Inside worker, waiting for assignemtn";
			HandleShardAssignment();
			network->barrier();
			worker_test_data();
		}
	}

	shutdown();
	return 0;
}


