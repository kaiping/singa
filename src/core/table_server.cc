//  Copyright © 2014 Anh Dinh. All Rights Reserved.

#include "core/table_server.h"
#include "utils/global_context.h"
#include "utils/network_thread.h"
#include "core/disk-table.h"



namespace lapis {
void TableServer::StartTableServer(const std::map<int, GlobalTable*>& tables) {
  VLOG(3)<<"start table server";
  tables_=tables;
  net_ = NetworkThread::Get();
  server_id_ = net_->id();
  VLOG(3)<<"table server id "<<server_id_;
  //  set itself as the current worker for the table
  for (auto& i: tables){
    VLOG(3)<<"in set worker";
    i.second->set_worker(this);
  }
  //  register to the manager
  RegisterWorkerRequest req;
  req.set_id(server_id_);
  VLOG(3)<<"before send msg to "<<GlobalContext::kCoordinatorRank;
  net_->Send(GlobalContext::kCoordinatorRank, MTYPE_REGISTER_WORKER, req);
  VLOG(3)<<"after send msg to "<<GlobalContext::kCoordinatorRank;
  // register callbacks
  net_->RegisterCallback(MTYPE_SHARD_ASSIGNMENT,
                         boost::bind(&TableServer::HandleShardAssignment, this));
  net_->RegisterCallback(MTYPE_DATA_PUT_REQUEST,
		  	  	  	  	 boost::bind(&TableServer::HandleDataPut, this));
  net_->RegisterCallback(MTYPE_DATA_PUT_REQUEST_FINISH,
		  	  	  	  	 boost::bind(&TableServer::FinishDataPut, this));

  net_->RegisterRequestHandler(MTYPE_PUT_REQUEST,
                               boost::bind(&TableServer::HandleUpdateRequest, this, _1));
  net_->RegisterRequestHandler(MTYPE_GET_REQUEST,
                               boost::bind(&TableServer::HandleGetRequest, this, _1));
}


void TableServer::HandleShardAssignment() {
  CHECK(GlobalContext::Get()->IsTableServer(id()))
    << "Assign table to wrong server " << id();
  ShardAssignmentRequest shard_req;
  net_->Read(GlobalContext::kCoordinatorRank, MTYPE_SHARD_ASSIGNMENT, &shard_req);
  //  request read from coordinator
  for (int i = 0; i < shard_req.assign_size(); i++) {
    const ShardAssignment &a = shard_req.assign(i);
    GlobalTable *t = tables_.at(a.table());
    t->get_partition_info(a.shard())->owner = a.new_worker();
    EmptyMessage empty;
    net_->Send(GlobalContext::kCoordinatorRank, MTYPE_SHARD_ASSIGNMENT_DONE, empty);
    LOG(INFO) << StringPrintf("Process %d is assigned shard (%d,%d)",
                              NetworkThread::Get()->id(), a.table(), a.shard());
  }
}


void TableServer::HandleDataPut(){
	DiskData data;
	net_->Read(GlobalContext::kCoordinatorRank, MTYPE_DATA_PUT_REQUEST, &data);
	(dynamic_cast<DiskTable*>(tables_.at(data.table())))->DumpToFile(&data);
}

void TableServer::FinishDataPut(){
	EmptyMessage msg;
	net_->Read(GlobalContext::kCoordinatorRank, MTYPE_DATA_PUT_REQUEST_FINISH, &msg);
	for (auto& t : tables_){
		(dynamic_cast<DiskTable*>(t.second))->finalize_data();
	}
	net_->Send(GlobalContext::kCoordinatorRank, MTYPE_DATA_PUT_REQUEST_DONE, msg);
}

//  respond to request
void TableServer::HandleGetRequest(const Message *message) {
  const HashGet *get_req = static_cast<const HashGet *>(message);
  TableData get_resp;
  // prepare
  get_resp.Clear();
  get_resp.set_source(server_id_);
  get_resp.set_table(get_req->table());
  get_resp.set_shard(-1);
  get_resp.set_done(true);
  get_resp.set_key(get_req->key());
  // fill data
  {
    GlobalTable *t = tables_.at(get_req->table());
    t->handle_get(*get_req, &get_resp);
  }
  net_->Send(get_req->source(), MTYPE_GET_RESPONSE, get_resp);
}

void TableServer::HandleUpdateRequest(const Message *message) {
  boost::recursive_mutex::scoped_lock sl(state_lock_);
  const TableData *put = static_cast<const TableData *>(message);
  GlobalTable *t = tables_.at(put->table());
  t->ApplyUpdates(*put);
}

int TableServer::peer_for_partition(int table, int shard) {
  return tables_[table]->owner(shard);
}
}
