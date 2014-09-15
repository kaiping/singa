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
void Coordinator::LoadData(const DataSourceProtos& sources,
                           const map<string, int>& stores) {
  // image file names, may from label source
  std::shared_ptr<std::vector<string>> filenames;
  for(auto source: sources){
    DataSource *ds=DataSourceFactory::Instance()->Create(source.type());
    filenames=ds->Init(source, filenames);
    int rid=0;
    const Shape &s=source.shape();
    FloatVector record;
    for(int i=0;i<s.width()*s.height()*s.channels();i++)
      record.add_data(0);
    VLOG(3)<<"start loading data "<<ds->name()<<" record size "<<record.data_size();
    while(!ds->eof()){
      ds->NextRecord(&record);
      mc_.PutData(stores.at(ds->name()), rid++, record);
    }
    VLOG(3)<<"flush data";
    mc_.FlushData(stores.at(ds->name()));
    VLOG(3)<<"finish loading data";
    delete ds;
  }
}


const StringIntMap Coordinator::CreateDataStores(
    const DataSourceProtos& sources, int fixed_server_id) {
  std::map<string, int> stores;
  for(auto& ds: sources){
    CHECK(stores.find(ds.name())==stores.end());
    stores[ds.name()]= mc_.CreateDataStore(ds.name(), fixed_server_id);
    VLOG(3) << "Created disk table with name " << ds.name();
  }
  return ToProtoMap(stores);
}

const DataStorageConfig Coordinator::CreateDataStorage(
    const DataProto& data){
  // create stores for train/validate/test data
  VLOG(3)<<"create data storages";
  DataStorageConfig conf;
  if(data.train_data_size()>0)
    conf.mutable_train_stores()->CopyFrom(CreateDataStores(data.train_data()));
  VLOG(3)<<"finish train stores";
  if(data.validation_data_size()>0)
    conf.mutable_val_stores()->CopyFrom(CreateDataStores(data.validation_data(), 0));
  VLOG(3)<<"finish val stores";
  if(data.test_data_size()>0)
    conf.mutable_test_stores()->CopyFrom(CreateDataStores(data.test_data()));
  VLOG(3)<<"finish test stores";
  conf.mutable_tables()->CopyFrom(ToProtoMap(mc_.GetDataStoreTable()));
  VLOG(3)<<"data storage finish";
  return conf;
}

const ParamStorageConfig Coordinator::CreateParamStorage() {
  VLOG(3)<<"create param storage";
  ParamStorageConfig config;
  int sid=mc_.CreateParamStore();
  StringIntMap *map=config.mutable_param_stores();
  StringIntPair *p=map->add_pair();;
  p->set_key("param");
  p->set_val(sid);
  config.mutable_tables()->CopyFrom(ToProtoMap(mc_.GetParamStoreTable()));
  return config;
}
void Coordinator::InitDistributedStorage(bool load_data, bool do_train,
    const ModelProto& model){
  VLOG(3)<<"setup storage";
  DistributedStorageConfig config;
    config.mutable_dsconfig()->CopyFrom(CreateDataStorage(model.data()));
  if(do_train)
    config.mutable_psconfig()->CopyFrom(CreateParamStorage());
  CHECK(config.has_dsconfig()||config.has_psconfig());
  VLOG(3)<<"send storage config";
  mpi_->Broadcast(MTYPE_STORAGE_CONFIG, config);
  VLOG(3)<<"after send storage config";
  // init table servers, must be after creating stores which creating tables
  InitTableServers(mc_.GetTables());
  VLOG(3)<<"tableserver assigned";

  if(load_data){
    LoadData(model.data().train_data(), ToStdMap(config.dsconfig().train_stores()));
    LoadData(model.data().validation_data(), ToStdMap(config.dsconfig().val_stores()));
    LoadData(model.data().test_data(), ToStdMap(config.dsconfig().test_stores()));
  }
  if(do_train){
    // TODO(Jingyang, Wei) model partition
    Net net(model.net());
    // setup the net, init parameters
    auto shapes=DataSource::ShapesOf(model.data().train_data());
    net.Setup(1, kAllocParam|kInitParam, shapes);
    mc_.Put(net.params());
  }
  VLOG(3)<<"finish init dist storage";
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

void Coordinator::RunOnCluster(const ModelProto& model) {
  VLOG(3)<<"start worker";
  mpi_->Broadcast(MTYPE_MODEL_CONFIG, model);
  bool *alive_workers=new bool[mpi_->size()];
  for(int i=0;i<mpi_->size();i++)
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

    Performance perf;
    if(mpi_->TryRead(MPI::ANY_SOURCE, MTYPE_PERFORMANCE, &perf, &src)) {
      LOG(INFO)<<FormatPerformance(src, perf);
    }
    Sleep(FLAGS_sleep_time);
  }
}

void Coordinator::Run(bool load_data, bool do_train) {
  ModelProto model;
  ReadProtoFromTextFile(context_->model_conf(), &model);
  if(do_train&&context_->standalone())
    RunStandalone(model);

  InitDistributedStorage(load_data, do_train, model);
  if(do_train&&!context_->standalone())
    RunOnCluster(model);
}
}  // namespace lapis
