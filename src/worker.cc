// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-06-28 16:01
#include <glog/logging.h>

#include "worker.h"
#include "model_controller/model.h"
#include "proto/model.pb.h"
#include "proto/common.pb.h"
#include "utils/proto_helper.h"

#include "net/net.h"
#include "utils/network_thread.h"
#include "utils/global_context.h"
#include "net/sgd_trainer.h"

namespace lapis {
Worker::Worker(){
  LOG(INFO) << "starting Worker...";
  mpi_=NetworkThread::Get();
  table_server_=nullptr;
}
Worker::~Worker() {
  Shutdown();
  if(table_server_!=nullptr)
    delete table_server_;
}

void SetupNet(const int batchsize,
              const char flag,
              Net *net,
              const DataSourceProtos& sources,
              const std::map<std::string, int> &store_map){
  auto shapes=DataSource::ShapesOf(sources);
  net->Setup(flag, batchsize, shapes, store_map);
}

bool Worker::ShouldIDoValidation(int step) {
  ShortMsg msg;
  msg.set_step(step);
  mpi_->Send(GlobalContext::kCoordinatorRank, MTYPE_VALIDATION, msg);
  mpi_->Read(GlobalContext::kCoordinatorRank, MTYPE_INSTRUCTION, &msg);
  return msg.answer();
}

const DistributedStorageConfig Worker::InitDistributedStorage(){
  DistributedStorageConfig config;
  VLOG(3)<<"read storage config";
  mpi_->Read(GlobalContext::kCoordinatorRank, MTYPE_STORAGE_CONFIG, &config);
  VLOG(3)<<"recv storage config";
  IntIntMap tables;
  CHECK(config.has_dsconfig()||config.has_psconfig());
  if(config.has_dsconfig())
    tables.MergeFrom(config.dsconfig().tables());
  VLOG(3)<<"merge data tables";
  if(config.has_psconfig())
    tables.MergeFrom(config.psconfig().tables());
  VLOG(3)<<"merge param tables";
  std::map<int, int> stdtables=ToStdMap(tables);
  model_controller_.CreateTables(stdtables);
  if(GlobalContext::Get()->AmITableServer()){
    table_server_=new TableServer();
    table_server_->StartTableServer(model_controller_.GetTables());
    VLOG(3)<<"table server tarted";
  }
  VLOG(3)<<"finish init storage";

  // Added by Anh, making sure all data is there
  mpi_->barrier();

  return config;
}
void Worker::Shutdown() {
	LOG(INFO) << "Worker is shutting down ...";
  mpi_->Flush();
  mpi_->Send(GlobalContext::kCoordinatorRank, MTYPE_WORKER_END, EmptyMessage());
  EmptyMessage msg;
  int src = 0;
  mpi_->Read(GlobalContext::kCoordinatorRank, MTYPE_WORKER_SHUTDOWN, &msg, &src);
  VLOG(3) << "Worker received MTYPE_WORKER_SHUTDOWN";
  table_server_->ShutdownTableServer();
  mpi_->Shutdown();
}

void Worker::Barrier(int step) {
  ShortMsg msg;
  msg.set_step(step);
  mpi_->Send(GlobalContext::kCoordinatorRank, MTYPE_BARRIER, msg);
  int src;
  // blocked until the coordinator reply, i.e., recv barriers from all workers
  mpi_->Read(GlobalContext::kCoordinatorRank, MTYPE_INSTRUCTION, &msg, &src);
}

void Worker::Run(bool load_data, bool do_train) {
  const DistributedStorageConfig config=InitDistributedStorage();
  if(!do_train){
    return;
  }
  std::map<std::string, int> train_stores, val_stores, test_stores;
  if(config.dsconfig().has_train_stores())
    train_stores=ToStdMap(config.dsconfig().train_stores());
  if(config.dsconfig().has_val_stores())
    val_stores=ToStdMap(config.dsconfig().val_stores());
  if(config.dsconfig().has_test_stores())
    test_stores=ToStdMap(config.dsconfig().test_stores());
  ModelProto model;
  mpi_->Read(GlobalContext::kCoordinatorRank, MTYPE_MODEL_CONFIG, &model);

  SGDTrainer trainer;
  trainer.Init(model.trainer(), &model_controller_);
  const SGDProto sgd=model.trainer().sgd();

  Net net(model.net());
  SetupNet(sgd.train_batchsize(), kAllocData|kAllocParam, &net,
          model.data().train_data(), train_stores);
  bool reset_net_for_training=false;
  std::vector<Param *> params = net.params();

  Performance perf;
  double comp_time=0.0, comm_time=0.0, sync_time=0.0;
  Timer clock;
  while (!trainer.HasFinished()) {
    perf.set_step(trainer.step());
    clock.reset();
    comm_time+=clock.elapsed();
    model_controller_.Get(params);
    if(trainer.ValidateNow()){
      if(ShouldIDoValidation(trainer.step())){
        VLOG(1)<<"start validation";
        // do validation
        if(sgd.validation_batchsize()!=sgd.train_batchsize()){
          SetupNet(sgd.validation_batchsize(), kAllocData, &net,
              model.data().validation_data(), val_stores);
          reset_net_for_training=true;
        }
        int nvalimgs=model.data().validation_data(0).shape().num();
        trainer.Validate(&net, &perf, nvalimgs/sgd.validation_batchsize());
        mpi_->Send(GlobalContext::kCoordinatorRank, MTYPE_PERFORMANCE, perf);
        // do test
        /*
           SetupNet(sgd.test_batchsize(), kAllocData, &net,
           model.data().test_data(), test_stores);
           trainer.Test(&net);
           */
      }
    }
    if(reset_net_for_training) {
      // workers should allocate memory for data and parameters. No need to
      // Init parameters, because they Get parameters from distributed table
      VLOG(3)<<"worker reset net for training"<<AllocData(kAllocData|kAllocParam);
      SetupNet(sgd.train_batchsize(), kAllocData|kAllocParam, &net,
          model.data().train_data(), train_stores);
      reset_net_for_training=false;
    }
    clock.reset();
    if(GlobalContext::Get()->synchronous())
      Barrier(trainer.step());
    sync_time+=clock.elapsed();
    clock.reset();
    trainer.TrainOneBatch(&net, &perf);
    comp_time+=clock.elapsed();
    clock.reset();
    mpi_->Send(GlobalContext::kCoordinatorRank, MTYPE_PERFORMANCE, perf);
    model_controller_.Update(params);
    comm_time+=clock.elapsed();
    VLOG(1)<<FormatTime(trainer.step(),comp_time, comm_time, sync_time);
  }
}
}  // namespace lapis
