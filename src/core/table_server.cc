//  Copyright © 2014 Anh Dinh. All Rights Reserved.

#include "core/table_server.h"
#include "utils/global_context.h"
#include "utils/network_thread.h"
#include "core/disk-table.h"
#include "core/request_dispatcher.h"

DECLARE_double(sleep_time);
namespace lapis {
TableServer::TableServer(){
}

void TableServer::StartTableServer(const std::map<int, GlobalTable*>& tables) {
  VLOG(3)<<"start table server";
  tables_=tables;
  net_ = NetworkThread::Get();
  server_id_ = net_->id();
  //  set itself as the current worker for the table
  for (auto& i: tables){
    i.second->set_worker(this);
  }
  //  register to the manager
  RegisterWorkerRequest req;
  req.set_id(server_id_);
  net_->Send(GlobalContext::kCoordinator, MTYPE_REGISTER_WORKER, req);

  // register callbacks
  // net_->RegisterCallback(MTYPE_SHARD_ASSIGNMENT,
    //                     boost::bind(&TableServer::HandleShardAssignment, this));

  // Start dispatcher
  VLOG(3) << "start request dispatchers from TableServer";
  RequestDispatcher *dispatcher = RequestDispatcher::Get();
  dispatcher->RegisterTableCallback(MTYPE_PUT_REQUEST,
                               boost::bind(&TableServer::HandlePutRequest, this, _1));
  dispatcher->RegisterTableCallback(MTYPE_UPDATE_REQUEST,
                                 boost::bind(&TableServer::HandleUpdateRequest, this, _1));
  dispatcher->RegisterTableCallback(MTYPE_GET_REQUEST,
                               boost::bind(&TableServer::HandleGetRequest, this, _1));
  dispatcher->RegisterDiskCallback(boost::bind(&TableServer::HandleDisk, this, _1));
  VLOG(3) << "done registering callback for dipsatcher";
}

void TableServer::ShutdownTableServer(){
	for (auto& i:tables_){
		vector<LogFile*> checkpoint_files = i.second->checkpoint_files(); 
		for (int i=0; i<checkpoint_files.size(); i++)
			delete checkpoint_files[i]; 
		checkpoint_files.clear(); 
	}
}

bool TableServer::HandleDisk(const Message* data){
	const DiskData *dt = static_cast<const DiskData*>(data);
	if (dt->is_empty())
		FinishDataPut(dt);
	else
		HandleDataPut(dt);
	return true;
}

void TableServer::HandleDataPut(const DiskData* data){
	for (auto& t : tables_){
		if ((int)data->table()==t.first)
			(dynamic_cast<DiskTable*>(t.second))->store(data);
	}
}

void TableServer::FinishDataPut(const DiskData* data) {
	for (auto& t : tables_) {
		if (t.first == (int) data->table()) {
			(dynamic_cast<DiskTable*>(t.second))->finalize_data();
		}
	}
  VLOG(3)<<"tb server finish";
	net_->Send(GlobalContext::kCoordinator, MTYPE_DATA_PUT_REQUEST_DONE,
			EmptyMessage());

  VLOG(3)<<"tb server finish send";
}

//  respond to request
bool TableServer::HandleGetRequest(const Message *message) {
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

    GlobalTable *t = tables_.at(get_req->table());
    if (t->handle_get(*get_req, &get_resp)){
    	net_->Send(get_req->source(), MTYPE_GET_RESPONSE, get_resp);
    	return true;
    }
    else return false;
}

bool TableServer::HandlePutRequest(const Message *message) {
  boost::recursive_mutex::scoped_lock sl(state_lock_);
  const TableData *put = static_cast<const TableData *>(message);
  GlobalTable *t = tables_.at(put->table());
  bool ret = t->ApplyPut(*put);
  return ret;
}

bool TableServer::HandleUpdateRequest(const Message *message) {
  boost::recursive_mutex::scoped_lock sl(state_lock_);
  const TableData *put = static_cast<const TableData *>(message);
  GlobalTable *t = tables_.at(put->table());
  bool ret = t->ApplyUpdates(*put);
  return ret;
}

int TableServer::peer_for_partition(int table, int shard) {
  return tables_[table]->owner(shard);
}
}
