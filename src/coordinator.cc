// Copyright © 2014 Wei Wang, Anh. All Rights Reserved.
// 2014-08-07 11:32
#include "coordinator.h"
#include "net/net.h"
#include "net/solver.h"
#include "utils/network_thread.h"
#include "utils/common.h"
#include "proto/common.pb.h"
#include "proto/model.pb.h"

using std::string;
using std::vector;
DECLARE_double(sleep_time);

namespace lapis {
Coordinator::Coordinator(const shared_ptr<GlobalContext>& gc){
  LOG(INFO)<<"Start coordinator...";
  context_=gc;
  mpi_=NetworkThread::Get();
}

Coordinator::~Coordinator() {
  for (auto* state: server_states_) {
    for (auto* taskid : state->local_shards)
      delete taskid;
    delete state;
  }
  Shutdown();
}

void Coordinator::InitTableServers(const std::map<int, GlobalTable*>& tables) {
  for (int i = context_->server_start();i<context_->server_end();++i){
    VLOG(3)<<"in table server "<<i;
    RegisterWorkerRequest req;
    int src = 0;
    VLOG(3)<<"before read msg ";
    mpi_->Read(MPI::ANY_SOURCE, MTYPE_REGISTER_WORKER, &req, &src);
    VLOG(3)<<"after read msg ";
    //  adding memory server.
    if (context_->IsTableServer(i)) {
      server_states_.push_back(new ServerState(i));
    }
  }
  LOG(INFO) << " All servers registered and started up";
  //  set itself as the current worker for the table
  for (auto &entry: tables)
    entry.second->worker_id_ = mpi_->id();

  // memory servers are specified in global context. Round-robin assignment
  int server_idx = 0;
  for (auto &entry: tables){
    VLOG(3)<<"num of shards"<<entry.second->num_shards()<<" for table"<< entry.first;
    int table=entry.first;
    for (int shard = 0; shard < entry.second->num_shards(); ++shard) {
      ServerState &server = *server_states_[server_idx];
      LOG(INFO) << "Assigning table ("<<table<<","<<shard<<") to server "
                <<server_states_[server_idx]->server_id;
      // TODO(Anh) may overwrite this field if #shards>#table_servers
      server.shard_id = shard;
      server.local_shards.insert(new TaskId(table, shard));
      server_idx = (server_idx + 1) % server_states_.size();
    }
  }
  VLOG(3)<<"table assignment";
  //  then send table assignment
  ShardAssignmentRequest req;
  for (size_t i = 0; i < server_states_.size(); ++i) {
    VLOG(3)<<"server states "<<i;
    ServerState &server = *server_states_[i];
    for (auto * task: server.local_shards) {
      ShardAssignment *s  = req.add_assign();
      s->set_new_worker(server.server_id);
      s->set_table(task->table);
      s->set_shard(task->shard);
      //  update local tables
      CHECK(tables.find(task->table)!=tables.end());
      GlobalTable *t = tables.at(task->table);
      VLOG(3)<<"table id"<<t->id();
      t->get_partition_info(task->shard)->owner = server.server_id;
      delete task;
    }
  }
  LOG(ERROR)<<"finish table assignment, req size "<<req.assign_size();
  mpi_->Broadcast(MTYPE_SHARD_ASSIGNMENT, req);
  mpi_->WaitForSync(MTYPE_SHARD_ASSIGNMENT_DONE, context_->num_table_servers());
  LOG(ERROR)<<"finish table server init";
}


//  wait for MTYPE_WORKER_END from other servers
//  send MTYPE_WORKER_SHUTDOWN messages to other
//  do not have to wait, simply exit.
void Coordinator::Shutdown() {
  /*
  EmptyMessage shutdown_msg;
  for (int i = 0; i < mpi_->size() - 1; i++) {
    mpi_->Send(i, MTYPE_SHUTDOWN, shutdown_msg);
  }
  mpi_->Flush();
  mpi_->Shutdown();
  */
}

Net* Coordinator::SetupNetShape(const ModelProto& model) {
  Net *net=new Net(model.net());
  // setup the net, init parameters
  int batchsize=model.solver().train_batchsize();
  vector<vector<int>> shapes;
  for(auto& shape: model.data().train_data().shape()){
    vector<int> s{batchsize};
    for(int k:shape.s())
      s.push_back(k);
    shapes.push_back(s);
  }
  net->InitDAryShape(shapes);
  return net;
}
// TODO model partitioning
const NetProto Coordinator::PartitionNet(Net* net){
  int pdim=0;
  for(Layer* layer: net->layers()){
    if(layer->name()=="pool5")
      pdim=1;
    if(layer->name()=="label")
      layer->SetPartition(-1);
    else if(layer->name()=="softmax")
      layer->SetPartition(-1);
    else
      layer->SetPartition(pdim);
  }
  NetProto netproto;
  net->ToProto(&netproto);
  return netproto;
}

void Coordinator::DistributePartition(const NetProto & netproto) {
  mpi_->Broadcast(MTYPE_NET_PARTITION, netproto);
}


void Coordinator::Start(const ModelProto& model) {
  SolverProto sp(model.solver());
  sp.mutable_sgd()->set_threshold(context_->num_groups());
  sp.mutable_adagrad()->set_threshold(0);
  TableDelegate* delegate=CreateTableDelegate(sp);
  InitTableServers(delegate->tables());

  Net* net=SetupNetShape(model);
  const NetProto partition=PartitionNet(net);
  delete net;
  /*
  int wid=0;
  for(auto& np: partitions){
    net->InitParameters();
    delegate->SplitParams(net->params(), wid);
    delegate->Put(net->params());
    delete net;
    wid++;
  }
  */
  DistributePartition(partition);
  EmptyMessage dummy_msg;
  LOG(ERROR)<<"Tell group 0 to init parameters";
  for(auto wid: context_->MembersOfGroup(0))
    mpi_->Send(wid, MTYPE_INIT_PARAMS, dummy_msg);
  int src;
  LOG(ERROR)<<"waiting Group 0 to finish init parameters";
  for(auto wid: context_->MembersOfGroup(0))
    mpi_->Read(wid, MTYPE_FINISH_INIT_PARAMS, &dummy_msg, &src);
  LOG(ERROR)<<"Group 0 has finished init parameters";
  Run();
}

void Coordinator::Resume() {
  /*
   * no need to fetch net partition from hdfs
   * workers will fetch netpartition
   * table server will resume from checkpoint
   * just notify workers to run
    TableDelegate* delegate=CreateTableDelegate(sp);
    InitTableServers(delegate->tables());
   */
  Run();
}

void Coordinator::Run(){
  LOG(ERROR)<<"Broadcast all workers to start running";
  EmptyMessage dummy_msg;
  mpi_->Broadcast(MTYPE_WORKER_START, dummy_msg);

  StateQueue<int> groups(context_->num_groups());
  int alive_workers=context_->num_groups()*context_->group_size();
  int src = 0;
  EmptyMessage end_msg;
  while(alive_workers) {
    if(mpi_->TryRead(groups.Next(),MTYPE_WORKER_END, &end_msg, &src)) {
      alive_workers--;
    }
    Performance perf;
    if(mpi_->TryRead(MPI::ANY_SOURCE, MTYPE_PERFORMANCE, &perf, &src)) {
      LOG(INFO)<<perf.ToString();
    }
    Sleep(FLAGS_sleep_time);
  }
  mpi_->Broadcast(MTYPE_SHUTDOWN, dummy_msg);
}

}  // namespace lapis
