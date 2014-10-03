// Copyright © 2014 Wei Wang, Anh. All Rights Reserved.
// 2014-08-07 11:32
#include "coordinator.h"
#include "proto/model.pb.h"
#include "net/net.h"
#include "net/sgd_trainer.h"
#include "model_controller/model.h"
#include "utils/network_thread.h"
#include "utils/proto_helper.h"
#include "datasource/data_source.h"
#include "utils/proto_helper.h"
#include "proto/common.pb.h"


using std::string;
DECLARE_double(sleep_time);

namespace lapis {
Coordinator::Coordinator() {
  LOG(INFO)<<"Start coordinator...";
  context_=GlobalContext::Get();
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
  for (int i = 0; i < context_->num_processes()-1; ++i) {
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
  VLOG(3)<<"finish table assignment, req size "<<req.assign_size();
  mpi_->SyncBroadcast(MTYPE_SHARD_ASSIGNMENT, MTYPE_SHARD_ASSIGNMENT_DONE, req);
  VLOG(3)<<"finish table server init";
}


//  wait for MTYPE_WORKER_END from other servers
//  send MTYPE_WORKER_SHUTDOWN messages to other
//  do not have to wait, simply exit.
void Coordinator::Shutdown() {
  EmptyMessage shutdown_msg;
  for (int i = 0; i < mpi_->size() - 1; i++) {
    mpi_->Send(i, MTYPE_WORKER_SHUTDOWN, shutdown_msg);
  }
  mpi_->Flush();
  mpi_->Shutdown();
}

// load all data sources (e.g., image and label) for either training data,
// vlidation data or test data.
void Coordinator::LoadData(const DataSourceProto& source, int tid){
  // image file names, may from label source
  DataSource *ds=DataSourceFactory::Instance()->Create(source.type());
  LOG(INFO)<<"Loading DataSource : "<<ds->name();
  ds->Init(source);
  int rid=0;
  ImageNetRecord record;
  while(!ds->eof()){
    ds->NextRecord(&record);
    delegate_->Put(tid, rid++, record);
    if(rid%10000==0)
      LOG(INFO)<<rid<<" records have been loaded";
  }
  delegate_->Flush(tid);
  delete ds;
}

void Coordinator::InitDistributedStorage(bool load_data, const DataProto& data){
  // init table servers, must be after creating stores which creating tables
  const std::map<int, GlobalTable*> tables=delegate_.CreateTables();
  InitTableServers(tables);

  if(load_data){
    if(data.has_train_data())
      LoadData(data.train_data(), tables[kTrain]);
    if(data.has_validation_data())
      LoadData(data.validation_data(), tables[kVal]);
    if(data.has_test_data())
      LoadData(data.test_data(), tables[kTest]);
  }
  VLOG(3)<<"finish init dist storage";
}

Net* Coordinator::SetupNetShape(const ModelProto& model) {
  Net net(model.net());
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

void Coordinator::FillParameterTable(int threshold,Solver* solver,  Net* net){
  if(solver->method()==SolverProto::kSGD){
    if(GlobalContext::Get()->synchronous())
      solver->sgd().set_threshold(threshold);
    delegate_->set_example(solver->sgd());
  }
  else{
    if(GlobalContext::Get()->synchronous())
      solver.adagrad().set_threshold(threshold);
    delegate_->set_example(solver.adagrad());
  }

  for(auto* param:net->params()){
    param->Fill();
    delegate_.Put(param);
    param->Free();
  }
}

bool Coordinator::DoValidationOn(int worker_id) {
  // TODO(wangwei) use policy to decide which worker to do validation
  return worker_id==0;
}
void Coordinator::RunStandalone(const ModelProto& model) {
  // workers should allocate memory for data and parameters. No need to
  // Init parameters, because they Get parameters from distributed table
  //trainer.Run(kAllocData|kAllocParam|kInitParam, &net);
}

// TODO model partitioning
const vector<ModelProto*> Coordinator::PartitionModel(const ModelProto& model, Net* net){
  ModelProto *proto=new ModelProto();
  proto->CopyFrom(model);
  proto->clear_net();
  NetProto *netproto=proto->mutable_net();
  net->ToProto(netproto);
  int batchsize=model.solver().train_batchsize();
  CHECK_EQ(batchsize%ngroups,0);
  vector<ModelProto*> ret{proto};
  return ret;
}

void Coordinator::DistributePartition(const GroupConfig& conf,
    const vectro<ModelProto*> & protos) {
  for(auto& group:conf.group()){
    CHECK_EQ(group.member().size(), protos.size());
    mpi_->Send(group.leader(), MTYPE_MODEL_CONFIG, *protos[0]);
    for (i = 1; i < protos.size(); i++) {
      mpi_->Send(group.member(i), MTYPE_MODEL_PARTITION, *protos[i]);
    }
  }
}

void Coordinator::RunOnCluster(const ModelProto& model) {
  VLOG(3)<<"start worker";
  int group_size=8;
  int ngroups=1;
  const GroupConfig conf=CreateGroups(group_size);
  mpi_->Broadcast(MTYPE_GROUP_CONFIG, config);
  Net* net=SetupNetShape(model);
  FillParameterTable(group_size, model.mutable_solver(), net);
  vector<ModelProto*> partitions=PartitionModel(model, net);
  DistributePartition(conf, parititions);
  StateQueue groups(ngroups);
  while(groups.HasValid()) {
    int src = 0;
    EmptyMessage end_msg;
    if(mpi_->TryRead(groups.Next(), end_msg, &src)) {
      groups.Invalide();
    }

    ShortMsg msg;
    if(mpi_->TryRead(MPI::ANY_SOURCE, MTYPE_VALIDATION, &msg, &src)) {
      if (DoValidationOn(src))
        msg.set_answer(true);
      else
        msg.set_answer(false);
      mpi_->Send(src, MTYPE_INSTRUCTION,msg);
    }

    PerformanceProto perf;
    if(mpi_->TryRead(MPI::ANY_SOURCE, MTYPE_PERFORMANCE, &perf, &src)) {
      LOG(INFO)<<FormatPerformance(perf);
    }
    Sleep(FLAGS_sleep_time);
  }
}
const GroupConfig Coordinator::CreateGroups(int group_size) {
  int nworkers=GlobalContext::Get()->num_workers();
  CHECK_EQ(nworkers%group_size, 0);
  GroupConfig config;
  // worker stats at rank 0
  int wrank=0;
  for (i = 0; i < nworkers/group_size; i++) {
    Group *group=config->add_group();
    group->set_leader(wrank++);
    for (k = 0; k < group_size; k++) {
      group->add_member(wrank++);
    }
  }
  return config;
}

void Coordinator::InitTableDelegate(const SolverProto& solver){
  if(solver().method()==SolverProto::kSGD){
    delegate_=new TypedTableDelegate<int, SGDValue>();
  }
  else{
    delegate_=new TypedTableDelegate<int, AdaGradValue>();
  }
}
void Coordinator::Run(bool load_data, bool do_train) {
  ModelProto model;
  ReadProtoFromTextFile(context_->model_conf(), &model);
  InitTableDelegate(model.solver(),8);
  InitDistributedStorage(load_data, model);
  if(do_train&&!context_->standalone())
    RunOnCluster(model);
}
}  // namespace lapis
