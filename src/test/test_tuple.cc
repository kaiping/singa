#include "proto/model.pb.h"
#include "proto/common.pb.h"
#include "utils/global_context.h"
#include "server.h"
#include "proto/worker.pb.h"
#include "utils/network_service.h"
#include "core/common.h"
#include "core/network_queue.h"
#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>
/**
 * @file test_tuple.cc
 *
 * Test performance of TableServer put/get/update operations.
 */
DECLARE_double(sleep_time);

using namespace lapis;
using namespace std;


#define NKEYS 500
#define TUPLE_SIZE 1000

void Put(int tid, int version) {
	RequestBase request;
	request.set_table(0);
	request.set_source(NetworkService::Get()->id());
	PutRequest *put_req = request.MutableExtension(PutRequest::name);
	int shard = tid % GlobalContext::Get()->num_servers();
	put_req->set_shard(shard);
	TableData *tuple = put_req->mutable_data();

	TKey* key = tuple->mutable_key();
	TVal* val = tuple->mutable_value();

	key->set_id(tid);
	key->set_version(version);

	DAryProto *data = val->mutable_data();
	for (int i = 0; i < TUPLE_SIZE; i++)
		data->add_value(0.0f);

	// TODO check the msg type
	NetworkService::Get()->Send(shard, MTYPE_REQUEST, request);
}

void Update(int tid, int version) {
	RequestBase request;
	request.set_table(0);
	request.set_source(NetworkService::Get()->id());
	UpdateRequest *update_req = request.MutableExtension(UpdateRequest::name);
	int shard = tid % GlobalContext::Get()->num_servers();
	update_req->set_shard(shard);
	TableData *tuple = update_req->mutable_data();

	TKey* key = tuple->mutable_key();
	TVal* val = tuple->mutable_value();

	key->set_id(tid);
	key->set_version(version);

	DAryProto *data = val->mutable_grad();
	for (int i = 0; i < TUPLE_SIZE; i++)
		data->add_value(1.0f);
	// TODO check the msg type
	NetworkService::Get()->Send(shard, MTYPE_REQUEST, request);
}

void print_result(TableData *data){
	TKey *key = data->mutable_key();
	TVal *val = data->mutable_value();
	int k = key->id();
	VLOG(3) << "key = " << k;
	string s;
	for (int i=0; i<TUPLE_SIZE; i++)
		s.append(to_string(val->mutable_data()->value(i))).append(" ");
	VLOG(3) << "val = " <<s;
}

void AsyncGet(int tid, int version) {
	RequestBase request;
	request.set_table(0);
	request.set_source(NetworkService::Get()->id());
	GetRequest *get_req = request.MutableExtension(GetRequest::name);
	int shard = tid % GlobalContext::Get()->num_servers();
	get_req->set_shard(shard);

	TKey *key = get_req->mutable_key();
	key->set_id(tid);
	key->set_version(version);
	NetworkService::Get()->Send(shard, MTYPE_REQUEST, request);

}

void Collect(){
	int count = NKEYS;
	while (count){
		while (true) {
				Message *resp = NetworkService::Get()->Receive();
				if (!resp)
					Sleep(FLAGS_sleep_time);
				else
					break;
			}
		count--;
	}
}

/**
 * Workers wait for the barrier, then one of them send SHUTDOWN message
 * to all table servers.
 */
void worker_send_shutdown(int id){
	auto gc = lapis::GlobalContext::Get();
	NetworkService *network_service_ = NetworkService::Get().get();
	EmptyMessage msg;
	MPI_Barrier(gc->mpicomm());
	if (gc->rank()==id){
		for (int i=0; i<gc->num_procs(); i++){
			if (gc->IsTableServer(i))
				network_service_->Send(0, MTYPE_SHUTDOWN,msg);
		}
	}
}

/**
 * One worker with the specific ID puts, others wait.
 */
void worker_load_data(int id){
	auto gc = lapis::GlobalContext::Get();
	if (gc->rank()==id)
		for (int i=0; i<NKEYS; i++)
			Put(i,0);
	MPI_Barrier(gc->mpicomm());
}

void worker_update_data(){
	for (int i=0; i<NKEYS; i++)
		Update(i,0);
	VLOG(3) << "Done update ...";
}

/*
 * Async get.
 */
void worker_get_data(){
	for (int i=0; i<NKEYS; i++)
		AsyncGet(i,0);
	VLOG(3) << "Done send get request ...";
	Collect();
	VLOG(3) << "Done collect ...";
}

void start_network_service_for_worker(){
	NetworkService *network_service_ = NetworkService::Get().get();
	network_service_->Init(GlobalContext::Get()->rank(), Network::Get().get(), new SimpleQueue());
	network_service_->StartNetworkService();
}

int main(int argc, char **argv) {
	google::InitGoogleLogging(argv[0]);
	gflags::ParseCommandLineFlags(&argc, &argv, true);

	int provided;


	MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);


	FLAGS_logtostderr = 1;


	// Init GlobalContext
	Cluster cluster;
	cluster.set_server_start(0);
	cluster.set_server_end(1);
	cluster.set_worker_start(1);
	cluster.set_worker_end(2);
	cluster.set_group_size(1);
	cluster.set_data_folder("/data1/wangwei/lapis");

	auto gc = lapis::GlobalContext::Get(cluster);


	// worker or table server
	if (gc->AmITableServer()) {
		lapis::TableServer server;
		SGDProto sgd;
		sgd.set_learning_rate(0.01);
		sgd.set_momentum(0.9);
		sgd.set_weight_decay(0.1);
		sgd.set_gamma(0.5);
		sgd.set_learning_rate_change_steps(1);
		server.Start(sgd);
	} else {
		start_network_service_for_worker();
		worker_load_data(cluster.worker_start());
		worker_update_data();
		worker_get_data();
		worker_update_data();
		worker_send_shutdown(cluster.worker_start());
		NetworkService::Get()->Shutdown();
	}
	gc->Finalize();
	MPI_Finalize();
	VLOG(3) << "End, process "<< gc->rank();
	return 0;
}

