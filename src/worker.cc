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
#include "core/table_server.h"

namespace lapis {
Worker::Worker(){
  LOG(INFO) << "starting Worker...";
  mpi_=NetworkThread::Get();
}

void Worker::SetupNet(const int batchsize,
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
std::map<std::string, int> ToStdMap(const StringIntMap& gmap) {
  std::map<std::string, int> stdmap;
  for(auto& pair: gmap.pair())
    stdmap[pair.key()]=pair.val();
  return stdmap;
}


std::map<int, int> ToStdMap(const IntIntMap& gmap) {
  std::map<int, int> stdmap;
  for(auto& pair: gmap.pair())
    stdmap[pair.key()]=pair.val();
  return stdmap;
}
void Worker::Run() {
  ModelProto model;
  mpi_->Read(GlobalContext::kCoordinatorRank, MTYPE_MODEL_CONFIG, &model);
  DistributedStorageConfig sconfig;
  mpi_->Read(GlobalContext::kCoordinatorRank, MTYPE_STORAGE_CONFIG, &sconfig);

  std::map<std::string, int> train_stores=ToStdMap(sconfig.train_stores());
  std::map<std::string, int> val_stores=ToStdMap(sconfig.val_stores());
  std::map<std::string, int> test_stores=ToStdMap(sconfig.test_stores());
  std::map<int, int> tables=ToStdMap(sconfig.tables());
  mc_.CreateTables(tables);
  TableServer *ts=nullptr;
  if(GlobalContext::Get()->AmITableServer()) {
    TableServer *ts=new TableServer();
    ts->StartTableServer(mc_.GetTables());
  }

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
            model.validation_data(), val_stores);
        trainer.Validate(&net);
        // do test
        SetupNet(sgd.test_batchsize(), kAllocData, &net,
            model.test_data(), test_stores);
        trainer.Test(&net);
        reset_net_for_training=true;
      }
    }
    if(reset_net_for_training) {
      // workers should allocate memory for data and parameters. No need to
      // Init parameters, because they Get parameters from distributed table
      SetupNet(sgd.train_batchsize(), kAllocData|kAllocParam, &net,
                model.train_data(), train_stores);
      reset_net_for_training=false;
    }
    trainer.TrainOneBatch(&net);
  }
  if(ts!=nullptr){
    ts->ShutdownTableServer();
    delete ts;
  }
}
}  // namespace lapis
