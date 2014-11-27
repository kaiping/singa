//  Copyright © 2014 Anh Dinh. All Rights Reserved.

#include "core/table_server.h"
#include "utils/global_context.h"
#include "utils/network_thread.h"
#include "core/disk-table.h"
#include "core/request_dispatcher.h"

DECLARE_double(sleep_time);

/**
 * @file table_server.cc
 * Implementation of TableServer class.
 * @see table_server.h.
 */
namespace lapis {

/**
 * Initialize TableServer.
 * First, it sets itself as the process for handling requests of the given tables
 * (i.e. its owner, but ownership will change after the shard assignment by the coordinator).
 *
 * Next, it initializes the request dispatcher (@see RequestDispatcher) and registers
 * its methods as callback to the request queue.
 *
 * Finally, it notifies the coordinator that it is ready to process requests.
 */
void TableServer::StartTableServer(const std::map<int, GlobalTable*>& tables) {

	// associate itself with the tables.
	tables_ = tables;
	net_ = NetworkThread::Get();
	server_id_ = net_->id();
	for (auto& i : tables) {
		i.second->set_worker(this);
	}

	// start dispatcher and register callbacks.
	RequestDispatcher *dispatcher = RequestDispatcher::Get();
	dispatcher->RegisterTableCallback(MTYPE_PUT_REQUEST,
			boost::bind(&TableServer::HandlePutRequest, this, _1));
	dispatcher->RegisterTableCallback(MTYPE_UPDATE_REQUEST,
			boost::bind(&TableServer::HandleUpdateRequest, this, _1));
	dispatcher->RegisterTableCallback(MTYPE_GET_REQUEST,
			boost::bind(&TableServer::HandleGetRequest, this, _1));

	// notify the coordinator.
	RegisterWorkerRequest req;
	req.set_id(server_id_);
	net_->Send(GlobalContext::kCoordinator, MTYPE_REGISTER_WORKER, req);
}

void TableServer::ShutdownTableServer() {
	for (auto& i : tables_) {
		map<int, LogFile*>* checkpoint_files = i.second->checkpoint_files();
		for (auto iterator = checkpoint_files->begin();
				iterator != checkpoint_files->end(); iterator++)
			delete iterator->second;
	}
}

bool TableServer::HandleGetRequest(const Message *message) {
	const HashGet *get_req = static_cast<const HashGet *>(message);
	TableData get_resp;
	get_resp.Clear();
	get_resp.set_source(server_id_);
	get_resp.set_table(get_req->table());
	get_resp.set_shard(-1);
	get_resp.set_done(true);
	get_resp.set_key(get_req->key());


	GlobalTable *t = tables_.at(get_req->table());
	if (t->HandleGet(*get_req, &get_resp)) {
		net_->Send(get_req->source(), MTYPE_GET_RESPONSE, get_resp);
		return true;
	}
	else return false;
}

bool TableServer::HandlePutRequest(const Message *message) {
	const TableData *put = static_cast<const TableData *>(message);
	GlobalTable *t = tables_.at(put->table());
	bool ret = t->ApplyPut(*put);
	return true;
}

bool TableServer::HandleUpdateRequest(const Message *message) {
	const TableData *put = static_cast<const TableData *>(message);
	GlobalTable *t = tables_.at(put->table());
	bool ret = t->ApplyUpdates(*put);
	return ret;
}

} //namespace lapis
