#include "proto/model.pb.h"
#include "proto/common.pb.h"
#include "utils/global_context.h"
#include "server.h"
#include "proto/worker.pb.h"
#include "utils/network_service.h"
#include "core/common.h"
#include "core/network_queue.h"
/**
 * @file test_tuple.cc
 *
 * Test performance of TableServer put/get/update operations.
 */
DECLARE_double(sleep_time);

using namespace lapis;

const int kTupleSize = 100;

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
	for (int i = 0; i < kTupleSize; i++)
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

	DAryProto *data = val->mutable_data();
	for (int i = 0; i < kTupleSize; i++)
		data->add_value(1.0f);

	// TODO check the msg type
	NetworkService::Get()->Send(shard, MTYPE_REQUEST, request);
}

void Get(int tid, int version) {
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

	Message *resp;
	while (true) {
		resp = NetworkService::Get()->Receive();
		if (!resp)
			Sleep(FLAGS_sleep_time);
		else
			break;
	}
	VLOG(3) << "Got RESULT ...";
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
	cluster.set_worker_end(3);
	cluster.set_group_size(2);
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
		VLOG(3) << "Started network service ...";
		Put(0,0);
		Get(0,0);
		worker_send_shutdown(cluster.worker_start());
		NetworkService::Get()->Shutdown();
	}
	VLOG(3) << "Finalizing GC, process " << gc->rank();
	gc->Finalize();
	VLOG(3) << "Finalizing MPI, process "<< gc->rank();
	MPI_Finalize();
	VLOG(3) << "done, process "<< gc->rank();
	return 0;
}

