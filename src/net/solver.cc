// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-14 14:28

#include <glog/logging.h>
#include <leveldb/db.h>
#include <vector>

#include "proto/model.pb.h"
#include "net/solver.h"

namespace lapis {
Phase Solver::phase=Phase::kTrain;
Solver::Solver(const SolverProto &proto) {
  pause_=false;
  //! if step_>0, then the trainer is restored from a checkpoint
  step_ = proto.checkpoint_step();
  checkpoint_after_steps_ = proto.checkpoint_after_steps();
  checkpoint_every_steps_ = proto.checkpoint_every_steps();
  //! last checkpoint step
  checkpoint_step_ = proto.checkpoint_step();
  display_after_steps_ = proto.display_after_steps();
  display_every_steps_ = proto.display_every_steps();
  validation_after_steps_ = proto.validation_after_steps();
  validation_every_steps_ = proto.validation_every_steps();

  train_steps_=proto.train_steps();
  validation_steps_=proto.validation_steps();
}

void Solver::Setup(TableDelegate* delegate, const DataProto& dp, const NetProto& np){
  net_=new Net(np);
  net_->Setup();
  delegate_->SplitParams(net_->params(), GlobalContext::Get()->worker_id());
  string shard_folder=GlobalContext::Get()->shard_folder();
  train_shard_=shard_folder+"/"+dp.train_data().name()+"-leveldb";
  val_shard_=shard_folder+"/"+dp.validation_data().name()+"-leveldb";
}

Solver::~Solver() {
  delete net_;
}
void Solver::ToProto(SolverProto *proto) {
  proto->set_checkpoint_after_steps(checkpoint_after_steps_);
  proto->set_checkpoint_every_steps(checkpoint_every_steps_);
  proto->set_checkpoint_step(checkpoint_step_);
  proto->set_display_after_steps(display_after_steps_);
  proto->set_display_every_steps(display_every_steps_);
}
leveldb::DB* Solver::OpenShard(string path) {
  leveldb::DB* db_temp;
  leveldb::Options options;
  options.create_if_missing = false;
  options.max_open_files = 100;
  LOG(INFO) << "Opening leveldb " << path;
  leveldb::Status status = leveldb::DB::Open(options, path, &db_temp);
  CHECK(status.ok()) << "Failed to open leveldb " << path << std::endl
    << status.ToString();
  return db_temp;
}
void* PrefetchData(void* context){
  PrefetchArg *parg=static_cast<PrefetchArg*>(context);
  leveldb::Iterator* iter=parg->iter;
  Net* net=parg->net;
  const DAry& input= net->input_layer(0)->GetData(nullptr);
  Range nrng=input.IndexRange(0);
  for(int n=0;n<nrng.first;n++){
    if(!iter->Valid())
      iter->SeekToFirst();
    iter->Next();
  }
  Record record;
  for(int n=0;n<nrng.second-nrng.first;++n){
    CHECK(iter);
    CHECK(iter->Valid());
    record.ParseFromString(iter->value().ToString());
    for(auto* layer:net->input_layer())
      layer->AddInputRecord(record);
  }
  for(int n=0;n<input.shape(0)-nrng.second;n++){
    if(!iter->Valid())
      iter->SeekToFirst();
    iter->Next();
  }
}
void Solver::Validate(){
  val_perf_.Reset();
  leveldb::DB* db=OpenShard(val_shard_);
  leveldb::Iterator* iter(db->NewIterator(leveldb::ReadOptions()));
  iter->SeekToFirst();
  PrefetchArg arg{iter, net_};

  pthread_create(&prefetch_thread_, NULL, &PrefetchData, &arg);
  Solver::phase=kValidation;
  Performance perf;
  for(int b=0;b<validation_steps_;b++){
    pthread_join(prefetch_thread_, NULL);
    for(auto* layer:net_->input_layer())
      layer->SetInputData(nullptr);
    pthread_create(&prefetch_thread_, NULL, &PrefetchData, &arg);
    val_perf_.Aggregate(ValidateOneBatch(net_));
  }
  pthread_join(prefetch_thread_, NULL);
  delete iter;
  delete db;
}


void Solver::Train(){
  leveldb::DB* db=OpenShard(train_shard_);
  leveldb::Iterator* iter(db->NewIterator(leveldb::ReadOptions()));
  iter->SeekToFirst();
  PrefetchArg arg{iter, net_};

  pthread_create(&prefetch_thread_, NULL, &PrefetchData, &arg);
  //Debug();
  while (!HasFinished()) {
    LOG(INFO)<<step();
    pthread_join(prefetch_thread_, NULL);
    for(auto* layer:net_->input_layer())
      layer->SetInputData(nullptr);
    train_perf_.Aggregate(TrainOneBatch(net_));
    pthread_create(&prefetch_thread_, NULL, &PrefetchData, &arg);
    if(DisplayNow()){
      Pause();
      while(DisplayNow()&&PauseNow())
        sleep(0.001);
    }
    if(ValidateNow()){
      Pause();
      while(ValidateNow()&&PauseNow())
        sleep(0.1);
    }else{
      IncStep();
    }
  }
  delete db;
  delete iter;
  pthread_join(prefetch_thread_, NULL);
}
Performance Solver::TrainOneBatch(Net *net){
  delegate_->AsyncGet(net->params(),step());
  auto layers=net->layers();
  for(auto* layer: layers){
    VLOG(3)<<layer->name();
    for(auto* param: layer->GetParams())
      delegate_->AsyncCollect(param, step());
    layer->ComputeFeature();
  }
  Performance perf=net->output_layer(0)->CalcPerf(true, false);
  for (auto layer = layers.rbegin(); layer != layers.rend(); layer++){
    VLOG(3)<<(*layer)->name();
    (*layer)->ComputeGradient();
    for(auto* param: (*layer)->GetParams())
      delegate_->Update(param, step());
  }
  IncStep();
  return perf;
}

Performance Solver::ValidateOneBatch(Net *net){
  delegate_->AsyncGet(net->params(), step());
  for(auto* layer: net->layers()){
    VLOG(3)<<layer->name();
    for(auto* param: layer->GetParams())
      delegate_->AsyncCollect(param, step());
    layer->ComputeFeature();
  }
  return net->output_layer(0)->CalcPerf(true, true);
}

//Performance Solver::Test(Net *net) { }

bool Solver::HasFinished(){
  if (step_ >= train_steps_)
    return true;
  else
    return false;
}

}  // namespace lapis
