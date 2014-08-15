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

using std::string;
DECLARE_double(sleep_time);

namespace lapis {
Coordinator::Coordinator() {
  context_=GlobalContext::Get();
  mpi_=NetworkThread::Get();
}

Coordinator::~Coordinator() {
  for (auto* state: server_states_) {
    for (auto* taskid : state->local_shards)
      delete taskid;
    delete state;
  }
}

void Coordinator::InitTableServers(const std::map<int, GlobalTable*>& tables) {
  for (int i = 0; i < context_->num_processes()-1; ++i) {
    VLOG(3)<<"in start table server "<<i;
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
    VLOG(3)<<"num of shards "<<entry.second->num_shards();
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
    ServerState &server = *server_states_[i];
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
  mpi_->SyncBroadcast(MTYPE_SHARD_ASSIGNMENT, MTYPE_SHARD_ASSIGNMENT_DONE, req);
}

void Coordinator::StartWorkers(ModelProto &model){
  VLOG(3)<<"start worker";
  mpi_->Broadcast(MTYPE_MODEL_CONFIG, model);
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
}

// load all data sources (e.g., image and label) for either training data,
// vlidation data or test data.
void Coordinator::LoadData(const DataSourceProtos& sources,
                           const map<string, int>& stores) {
  // image file names, may from label source
  std::shared_ptr<std::vector<string>> filenames;
  for(auto source: sources){
    DataSource *ds=DataSourceFactory::Instance()->Create(source.type());
    filenames=ds->Init(ds_proto, filenames);
    int rid=0;
    Shape s(source.shape());
    s.set_num(1);
    FloatVector record;
    while(!ds->eof()){
      ds->NextRecord(&record);
      mc.PutData(stores.at(ds->name()), rid++, record);
    }
    delete ds;
  }
}

string FormatPerformance(const PerformanceProto& perf) {
  stringstream ss;
  if (perf.has_precision())
    ss<<StringPrintf("Precision %.3f, ", perf.precision());
  if (perf.has_recall())
    ss<<StringPrintf("Recall %.3f, ", perf.recall());
  if (perf.has_recall())
    ss<<StringPrintf("MAP %.3f ", perf.map());
  if (perf.has_recall())
    ss<<StringPrintf("Precision@50 %.3f ", perf.precision50());
  return ss.str();
}

std::map<string, int> Coordinator::CreateDataStores(
    const DataSourceProtos& sources) {
  std::map<string, int> stores;
  for(auto& ds: sources)
    stores[ds.name()]= =mc_->CreateDataStore();
  return stores;
}

void Coordinator::InitCluster(const ModelProto& model, Net* net){
  // create stores for train/validate/test data
  std::map<string, int> train_stores=CreateDataStores(model.train_data());
  std::map<string, int> val_stores=CreateDataStores(model.validate_data());
  std::map<string, int> test_stores=CreateDataStores(model.test_data());
  mc_->CreateParamStore();
  StorageConfig sconfig;
  sconfig.set_allocated_train_store(ToGoogleMap<string, int,
      StringIntMap, StringIntPair>(train_stores));
  sconfig.set_allocated_val_store(ToGoogleMap<string, int,
      StringIntMap, StringIntPair>(val_stores));
  sconfig.set_allocated_test_store(ToGoogleMap<string, int,
      StringIntMap, StringIntPair>(test_stores));
  sconfig.set_allocated_table(
      ToGoogleMap<int, int, IntIntMap, IntIntPair>(mc_->GetStoreTableMap());
  mpi_->Broadcast(MTYPE_STORAGE_CONFIG, sconfig);

  // init table servers, must be after creating stores which creating tables
  InitTableServers(mc_->tables());
  VLOG(3)<<"init table server finish";

  LoadData(model.train_data(), train_stores);
  LoadData(model.validation_data(), val_stores);
  LoadData(model.test_data(), test_stores);

  // setup the net, init parameters
  int batchsize=model.trainer().sgd().train_batchsize();
  auto shapes=DataSource::ShapesOf(train_sources());
  net.Setup(batchsize, kAllocParam|kInitParam, shapes, train_stores);
  // TODO(Jingyang, Wei) model partition
  mc_>Put(net.params());
}

void Coordinator::RunOnCluster(const ModelProto& model, Net *net) {
  InitCluster(model,net);
  StartWorkers(model);
  bool *alive_workers=new bool[mpi_->size()];
  for(int =0;i<mpi_->size();i++)
    alive_workers[i]=true;
  int num_alives=mpi_->size();
  while(alive_workers>0) {
    int src = 0;

    EmptyMessage end_msg;
    if(mpi_->TryRead(MPI::ANY_SOURCE, MTYPE_WORKER_END, &end_msg, &src)) {
      alive_workers[src]=false;
      num_alives--;
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
  Shutdown();
}

void Coordinator::RunStandalone(const ModelProto& model, Net *net) {
  // workers should allocate memory for data and parameters. No need to
  // Init parameters, because they Get parameters from distributed table
  //trainer.Run(kAllocData|kAllocParam|kInitParam, &net);
}
void Coordinator::Run() {
  ModelProto model;
  ReadProtoFromTextFile(context_->model_conf(), &model);
  Net net(model.net());
  mc_.Init();

  if(context_->standalone()){
    RunStandalone(model, &net);
  }else {
    RunOnCluster(model, &net);
  }
}
}  // namespace lapis
