//  Copyright © 2014 Anh Dinh. All Rights Reserved.

#include "core/memory-server.h"
#include "table-registry.h"
#include "global_context.h"

namespace lapis{
	MemoryServer::MemoryServer(){
		net_ = NetworkThread::Get();
		server_id_ = net_->id();
		manager_id_ = net_->size()-1;

		//  set itself as the current worker for the table
		TableRegistry::Map &t = TableRegistry::Get()->tables();
		for (TableRegistry::Map::iterator i = t.begin(); i != t.end(); ++i) {
		    i->second->set_worker(this);
		}

		//  register to the manager
		 RegisterWorkerRequest req;
		 req.set_id(server_id_);
		 network_->Send(manager_id_, MTYPE_REGISTER_WORKER, req);

		// register callbacks
		NetworkThread::Get()->RegisterCallback(MTYPE_SHARD_ASSIGNMENT,
		                                        boost::bind(&MemoryServer::HandleShardAssignment, this));
		NetworkThread::Get()->RegisterRequestHandler(MTYPE_PUT_REQUEST,
		                                        boost::bind(&MemoryServer::HandleUpdateRequest, this));
		NetworkThread::Get()->RegisterRequestHandler(MTYPE_GET_ReQUEST,
												boost::bind(&MemoryServer::HandleGetRequest, this));
	}


	void MemoryServer::HandleShardAssignment(){
		CHECK(GlobalContext::Get()->IsRoleOf(Role.kMemoryServer, id())) << "Assign table to wrong server " << id();

		ShardAssignmentRequest shard_req;
		while (net_->Read(manager_id_, MTYPE_SHARD_ASSIGNMENT, &shard_req)) {
			//  request read from DistributedMemoryManager
			for (int i = 0; i < shard_req.assign_size(); ++i) {
				const ShardAssignment &a = shard_req.assign(i);
				GlobalTable *t = TableRegistry::Get()->table(a.table());

				t->get_partition_info(a.shard())->owner = a.new_worker();

				EmptyMessage empty;
				network_->Send(manager_id_, MTYPE_SHARD_ASSIGNMENT_DONE, empty);
			}
	}

	//  respond to request
	void MemoryServer::HandleGetRequest(const Message* message){
		HashGet* get_req = static_cast<HashGet*>(message);
		Tabledata get_resp;

		// prepare
		get_resp.Clear();
		get_resp.set_source(server_id_);
		get_resp.set_table(get_req->table());
		get_resp.set_shard(-1);
		get_resp.set_done(true);

		// fill data
		{
			GlobalTable * t = TableRegistry::Get()->table(get_req->table());
			t->handle_get(*get_req, &get_resp);
		}

		net_->Send(get_req->source(), MTYPE_GET_RESPONSE, get_resp);
	}

	void MemoryServer::HandleUpdateRequest(const Message* message){
		boost::recursive_mutex::scoped_lock sl(state_lock_);

		TableData* put = static_cast<TableData*>(message);

		GlobalTable *t = TableRegistry::Get()->table(put.table());
		t->ApplyUpdates(*put);
	}

	int MemoryServer::peer_for_partition(int table, int shard){
		return TableRegistry::Get()->tables()[table]->owner(shard);
	}
}
