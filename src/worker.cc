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
  net->Setup(batchsize, flag, shapes, store_map);
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
  mpi_->Read(GlobalContext::kCoordinatorRank, MTYPE_STORAGE_CONFIG, &config);
  IntIntMap tables;
  tables.CopyFrom(config.dsconfig().tables());
  tables.MergeFrom(config.psconfig().tables());
  std::map<int, int> stdtables=ToStdMap(tables);
  mc_.CreateTables(stdtables);
  if(GlobalContext::Get()->AmITableServer()){
    table_server_=new TableServer();
    table_server_->StartTableServer(mc_.GetTables());
  }
  return config;
}
void Worker::Shutdown() {
  mpi_->Flush();
  mpi_->Send(GlobalContext::kCoordinatorRank, MTYPE_WORKER_END, EmptyMessage());
  EmptyMessage msg;
  int src = 0;
  mpi_->Read(GlobalContext::kCoordinatorRank, MTYPE_WORKER_SHUTDOWN, &msg, &src);
  mpi_->Shutdown();
}

void Worker::Run(bool load_data, bool do_train) {
  const DistributedStorageConfig config=InitDistributedStorage();
  if(!do_train){
      return;
  }
  ModelProto model;
  mpi_->Read(GlobalContext::kCoordinatorRank, MTYPE_MODEL_CONFIG, &model);
  std::map<std::string, int> train_stores=ToStdMap(config.dsconfig().train_stores());
  std::map<std::string, int> val_stores=ToStdMap(config.dsconfig().val_stores());
  std::map<std::string, int> test_stores=ToStdMap(config.dsconfig().test_stores());
  Net net(model.net());
  SGDTrainer trainer;
  trainer.Init(model.trainer(), &mc_);
  const SGDProto sgd=model.trainer().sgd();
  bool reset_net_for_training=true;
  while (!trainer.HasFinished()) {
    LOG(INFO)<<trainer.step();
    if(trainer.ValidateNow()){
      if(ShouldIDoValidation(trainer.step())){
        // do validation
        SetupNet(sgd.validation_batchsize(), kAllocData, &net,
            model.data().validation_data(), val_stores);
        trainer.Validate(&net);
        // do test
        SetupNet(sgd.test_batchsize(), kAllocData, &net,
            model.data().test_data(), test_stores);
        trainer.Test(&net);
        reset_net_for_training=true;
      }
    }
    if(reset_net_for_training) {
      // workers should allocate memory for data and parameters. No need to
      // Init parameters, because they Get parameters from distributed table
      SetupNet(sgd.train_batchsize(), kAllocData|kAllocParam, &net,
                model.data().train_data(), train_stores);
      reset_net_for_training=false;
    }
    trainer.TrainOneBatch(&net);
  }
}
}  // namespace lapis
