// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-14 14:28

#include <glog/logging.h>
#include <leveldb/db.h>
#include <vector>

#include "proto/model.pb.h"
#include "net/solver.h"
#include "da/gary.h"

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
  context_=GlobalContext::Get();
  mpi_=NetworkThread::Get();

}

void Solver::Setup(TableDelegate* delegate, const DataProto& dp, const NetProto& np){
  net_=new Net(np);
  net_->Setup();
  auto params=net_->params();
  auto grp_rank=context_->worker_id();
  delegate_=delegate;
  delegate_->SplitParams(params, grp_rank);
  string shard_folder=GlobalContext::Get()->shard_folder();
  train_shard_=shard_folder+"/"+dp.train_data().name()+"-leveldb";
  val_shard_=shard_folder+"/"+dp.validation_data().name()+"-leveldb";
}

void Solver::InitParams(){
  net_->InitParameters();
  delegate_->Put(net_->params());
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
  CHECK(iter);
  Net* net=parg->net;
  const DAry& input= net->input_layer(0)->GetData(nullptr);
  Range nrng=input.IndexRange(0);
  for(int n=0;n<nrng.first;n++){
    if(!iter->Valid()){
      LOG(INFO)<<"reset to start leveldb";
      iter->SeekToFirst();
    }
    iter->Next();
  }
  Record record;
  for(int n=0;n<nrng.second-nrng.first;++n){
    if(!iter->Valid()){
      LOG(INFO)<<"reset to start leveldb";
      iter->SeekToFirst();
    }
    record.ParseFromString(iter->value().ToString());
    //LOG(INFO)<<iter->key().ToString()<<" "<<record.label();
    for(auto* layer:net->input_layer())
      layer->AddInputRecord(record);
    iter->Next();
  }
  for(int n=0;n<input.shape(0)-nrng.second;n++){
    if(!iter->Valid()){
      LOG(INFO)<<"reset to start leveldb";
      iter->SeekToFirst();
    }
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

void Solver::ReportPerformance(Performance perf) {
  if(context_->AmIGroupLeader()){
    int toRcv=context_->group_size()-1;
    while(toRcv>0){
      Performance p;
      if(mpi_->TryRead(0, MTYPE_PERFORMANCE, &p)){
        perf.Aggregate(p);
        toRcv--;
      }
    }
    //context_->Send(GlobalContext::kCoordinator, MTYPE_PERFORMANCE, perf);
    LOG(ERROR)<<perf.ToString();
  }else{
    mpi_->Send(context_->leader(), MTYPE_PERFORMANCE, perf);
 }
}

void Solver::Train(){
  leveldb::DB* db=OpenShard(train_shard_);
  leveldb::Iterator* iter(db->NewIterator(leveldb::ReadOptions()));
  iter->SeekToFirst();
  PrefetchArg arg{iter, net_};

  //pthread_create(&prefetch_thread_, NULL, &PrefetchData, &arg);
  while (!HasFinished()) {
    //pthread_join(prefetch_thread_, NULL);
    PrefetchData(&arg);
    for(auto* layer:net_->input_layer())
      layer->SetInputData(nullptr);
    train_perf_.Aggregate(TrainOneBatch(net_));
    //pthread_create(&prefetch_thread_, NULL, &PrefetchData, &arg);
    if(DisplayNow()){
      ReportPerformance(train_perf_.Avg());
      train_perf_.Reset();
    }
    if(ValidateNow()){
      Validate();
      ReportPerformance(val_perf_.Avg());
    }
    IncStep();
  }
  delete db;
  delete iter;
  pthread_join(prefetch_thread_, NULL);
}
Performance Solver::TrainOneBatch(Net *net){
  LOG(ERROR)<<"Training Step : "<<step();
  delegate_->AsyncGet(net->params(),step());
  auto layers=net->layers();
  for(auto* layer: layers){
    for(auto* param: layer->GetParams()){
      delegate_->AsyncCollect(param, step());
      LOG(INFO)<<"param "<<param->name()<<" "<<param->data().Norm1();
    }
    layer->ComputeFeature();
    LOG(ERROR)<<layer->name()<<" norm "<<layer->data().Norm1();
  }
  Performance perf=net->output_layer(0)->CalcPerf(true, false);
  for (auto layer = layers.rbegin(); layer != layers.rend(); layer++){
    LOG(ERROR)<<(*layer)->name();
    (*layer)->ComputeGradient();
    for(auto* param: (*layer)->GetParams())
      delegate_->Update(param, step());
  }
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

void Solver::TimeOneBatch(int runs) {
  delegate_->Get(net_->params(), 0);
  Timer t;
  LOG(INFO)<<"Forwarding...";

  auto layers=net_->layers();
  for (auto* layer : layers){
    t.reset();
    for (int i = 0; i < runs; i++)
      layer->ComputeFeature();
    LOG(INFO)<<layer->name() <<": "<<t.elapsed()*1.0/runs;
    if(layer->name().find("conv")!=std::string::npos){
      ConvLayer* cl=dynamic_cast<ConvLayer*> (layer);
      LOG(INFO)<<"img2col "<<cl->img2col<<" col2img "<<cl->col2img
        <<" tdot "<<cl->tdot<<" tadd "<<cl->tadd;
    }
  }
  LOG(INFO)<<"Backwarding...";
  for (auto layer = layers.rbegin(); layer != layers.rend(); layer++){
    t.reset();
    for(int i=0;i<runs;i++)
      (*layer)->ComputeGradient();
    LOG(INFO)<<(*layer)->name()<<": "<<t.elapsed()*1.0/runs;
    if((*layer)->name().find("conv")!=std::string::npos){
      ConvLayer* cl=dynamic_cast<ConvLayer*> (*layer);
      LOG(INFO)<<"img2col "<<cl->img2col<<" col2img "<<cl->col2img
        <<" tdot "<<cl->tdot<<" tadd "<<cl->tadd;
    }
  }
}


//Performance Solver::Test(Net *net) { }

bool Solver::HasFinished(){
  if (step_ >= train_steps_)
    return true;
  else
    return false;
}

}  // namespace lapis
