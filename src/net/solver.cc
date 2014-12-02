// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-14 14:28

#include <glog/logging.h>
#include <mpi.h>
#include <vector>
#include "utils/proto_helper.h"
#include "proto/model.pb.h"
#include "net/solver.h"
#include "da/gary.h"
#include "utils/debug.h"

namespace lapis {
Phase Solver::phase=Phase::kTrain;
Solver::Solver(const SolverProto &proto) {
  //! if step_>0, then the trainer is restored from a checkpoint
  step_ = proto.step();
  proto_=proto;
  /*
  display_after_steps_ = proto.display_after_steps();
  display_every_steps_ = proto.display_every_steps();
  validation_after_steps_ = proto.validation_after_steps();
  validation_every_steps_ = proto.validation_every_steps();
  test_after_steps_ = proto.test_after_steps();
  test_every_steps_ = proto.test_every_steps();

  batchsize_=proto.batchsize();
  train_steps_=proto.train_steps();
  test_steps_=proto.test_steps();
  validation_steps_=proto.validation_steps();
  sgd_=proto.sgd();
  */
  context_=GlobalContext::Get();

  string data_folder=GlobalContext::Get()->data_folder();
  train_shard_=data_folder+"/train";
  val_shard_=data_folder+"/validation";
  test_shard_=data_folder+"/test";
}
Solver::~Solver() {
  delete net_;
  delete delegate_;
}

/*
void Solver::LocalUpdate(Param* param, int step) {
  float lr=UpdateHyperParam(step, sgd_.learning_rate_change(),
      sgd_.learning_rate_change_steps(),sgd_.base_learning_rate(), sgd_.gamma());
  DAry* history=param->mutable_history();
  const DAry& grad=param->grad();
  const DAry& data=param->data();
  int len=data.local_size();
  CHECK_EQ(len, grad.local_size());
  CHECK_EQ(len, history->local_size());
  lr=lr*param->learning_rate_multiplier();
  float w=sgd_.weight_decay()*param->weight_decay_multiplier();
  // hist=hist-lr*grad
  DAry::arymath().madd(history->dptr(), lr, grad.dptr(), history->dptr(), len);
  // hist=hist-lr*weight*param
  if(w>0)
    DAry::arymath().madd(history->dptr(), lr*w, data.dptr(), history->dptr(), len);

  // param+=history/n, /data->n_update()
  DAry::arymath().sub(data.dptr(), data.dptr(), history->dptr(), len);
  // hist=hist*mom
  DAry::arymath().mul(history->dptr(), sgd_.momentum(), history->dptr(), len);
}
*/

void Solver::Setup(const NetProto& np){
  net_=SetupNeuralNet(np);
  auto params=net_->params();
  auto grp_rank=context_->worker_id();
  delegate_=new TableDelegate();
  delegate_->SplitParams(params, grp_rank);
}

Net* Solver::SetupNeuralNet(const NetProto& proto) {
  Net *net=new Net(proto);
  Shard shard(train_shard_, Shard::kRead);
  Record record;
  string key;
  shard.Next(&key, &record);
  // setup the net, init parameters
  net->SetNetShape(proto_.batchsize(), record);

  if(proto_.partition()==SolverProto::kHybrid){
    int pdim=0;
    for(Layer* layer: net->layers()){
      if(layer->name()=="fc6")
        pdim=1;
      if(layer->name()=="fc8")
        pdim=0;
      layer->SetupDAry(pdim);
    }
  }else if (proto_.partition()==SolverProto::kData){
    for(Layer* layer: net->layers())
      layer->SetupDAry(0);
  }else{
     for(Layer* layer: net->layers()){
      if(layer->name()=="softmax")
        layer->SetupDAry(-1);
      else
        layer->SetupDAry(1);
     }
  }
  // data are envenly distributed to all workers, the input layer must be
  // partitioned on num (0-th) dim
  // fc8 and imgcol1's 1-th dim mode 2^k !=0
  for(Layer* layer: net->layers()){
    if(layer->HasInput()||layer->name()=="fc8"||layer->name()=="imgcol1")
      layer->SetupDAry(0);
  }
  // net->AllocMemory();
  return net;
}
void Solver::InitParams(){
  for(auto* param: net_->params()){
    param->Fill();
  }
  for(auto* param: net_->params())
    if(!param->partition()||context_->num_groups()>1)
      delegate_->Put(param);
}
void Solver::ToProto(SolverProto *proto) {
  /*
  proto->set_checkpoint_after_steps(checkpoint_after_steps_);
  proto->set_checkpoint_every_steps(checkpoint_every_steps_);
  proto->set_checkpoint_step(checkpoint_step_);

  proto->set_display_after_steps(display_after_steps_);
  proto->set_display_every_steps(display_every_steps_);

  proto->set_validation_after_steps(validation_after_steps_);
  proto->set_validation_every_steps(validation_every_steps_);

  proto->set_test_after_steps(test_after_steps_);
  proto->set_test_every_steps(test_every_steps_);
  */
  proto->MergeFrom(proto_);
  proto->set_step(step_);
}
void debug_mem(string prefix){
  char buf[1024];
  sprintf(buf, "%30s, %12lu", prefix.c_str(), getCurrentRSS());
  LOG(INFO)<<string(buf);
}

Performance Solver::Test(const Phase& phase){
  string shard;
  int nsteps;
  if(phase==Phase::kValidation){
    shard=val_shard_;
    nsteps=proto_.validation_steps();
  }
  else if(phase==Phase::kTest){
    shard=test_shard_;
    nsteps=proto_.test_steps();
  }
  else
    LOG(ERROR)<<"Phase must be kValidation or kTest";
  // fetch params once

  Prefetcher prefetcher(shard, net_);
  std::thread *thd=nullptr;
  thd=new std::thread(std::ref(prefetcher));
  Solver::phase=phase;
  Performance perf;
  for(int b=0;b<nsteps;b++){
    thd->join();
    delete thd;
    for(auto* layer:net_->input_layer())
      layer->SetInputData(nullptr);
    thd=new std::thread(std::ref(prefetcher));
    perf.Aggregate(TestOneBatch(net_, step_));
  }
  thd->join();
  delete thd;
  /*
  for(auto* layer:net_->input_layer())
    layer->SetInputData(nullptr);
    */
  Solver::phase=Phase::kTrain;
  return perf;
}

void Solver::ReportPerformance(string prefix, Performance perf) {
  LOG(ERROR)<<"Train Step: "<<step_<<" "<<prefix<<" "<<perf.ToString();
}

void Solver::Train(int start_step){
  step_=start_step;
  Prefetcher prefetcher(train_shard_, net_);
  std::thread *thd=nullptr;
  thd=new std::thread(std::ref(prefetcher));
  while (!HasFinished()) {
    Solver::phase=Phase::kTrain;
    thd->join();
    delete thd;
    for(auto* layer:net_->input_layer())
      layer->SetInputData(nullptr);
    if(!ValidateNow()&&!TestNow())
      thd=new std::thread(std::ref(prefetcher));
    train_perf_.Aggregate(TrainOneBatch(net_, step_));
    if(DisplayNow()){
      ReportPerformance("Train", train_perf_.Avg());
      DebugInfo(net_);
      train_perf_.Reset();
    }
    if(ValidateNow()){
      Performance perf=Test(Phase::kValidation);
      ReportPerformance("Val  ", perf.Avg());
    }
    if(TestNow()){
      Performance perf=Test(Phase::kTest);
      ReportPerformance("Test ", perf.Avg());
    }
    if(ValidateNow()||TestNow())
      thd=new std::thread(std::ref(prefetcher));
    IncStep();
  }
  thd->join();
  delete thd;
}
/*
void Solver::DoLocalCheckpoint(Net* net){
  for(auto* param: net->params()){
    if(param->partition()&&GlobalContext::Get()->num_groups()==1){
      DAryProto data;
      param->data().ToProto(&data, true);
      DAryProto grad;
      param->history().ToProto(&grad, true);
      char fname[256];
      sprintf(fname, "%s/local_cp/param_%d_%d.dat", context_->data_folder().c_str(),
          param->id(), step_);
      WriteProtoToBinaryFile(data, fname);
      sprintf(fname, "%s/local_cp/param_%d_%d.hst", context_->data_folder().c_str(),
          param->id(), step_);
      WriteProtoToBinaryFile(grad, fname);
    }
  }
}
*/
void Solver::DebugInfo(Net* net){
  char display[4096];
  auto layers=net->layers();
  LOG(INFO)<<"Train Step: "<<step_;
  for(auto* layer: layers){
    sprintf(display, "Forward layer  %10s data norm1 %13.9f",
        layer->name().c_str(), layer->data().Norm1());
    LOG(INFO)<<string(display);
  }
  for (auto layer = layers.rbegin(); layer != layers.rend(); layer++){
    sprintf(display, "Backward layer %10s grad norm1 %13.9f",
        (*layer)->name().c_str(), (*layer)->grad().Norm1());
    LOG(INFO)<<string(display);
  }
  for(auto* layer: layers){
    for(auto* param: layer->GetParams()){
      sprintf(display, "Layer %10s, param id %2d, name %10s, value norm1 %13.9f , grad norm1 %13.9f",
          layer->name().c_str(), param->id(),
          param->name().c_str(), param->data().Norm1(), param->grad().Norm1());
      LOG(INFO)<<string(display);
    }
  }
}



Performance Solver::TrainOneBatch(Net *net, int step){
  auto layers=net->layers();
  if(step==0){
    for(auto* param: net->params()){
        delegate_->AsyncGet(param,step);
    }
  }
  for(auto* layer: layers){
    for(auto* param: layer->GetParams()){
        delegate_->AsyncCollect(param, step);
    }
    if(layer->PreSyncF())
      MPI_Barrier(context_->mpicomm());
    layer->ComputeFeature();
    if(layer->PostSyncF())
      MPI_Barrier(context_->mpicomm());
  }
  Performance perf=net->output_layer(0)->CalcPerf(true, false);
  for (auto layer = layers.rbegin(); layer != layers.rend(); layer++){
    if((*layer)->PreSyncG())
      MPI_Barrier(context_->mpicomm());
    (*layer)->ComputeGradient();
    for(auto* param: (*layer)->GetParams()){
        delegate_->Update(param, step);
        delegate_->AsyncGet(param,step+1);
    }
    if((*layer)->PostSyncG())
      MPI_Barrier(context_->mpicomm());
  }
  //DebugInfo(net);
  return perf;
}

Performance Solver::TestOneBatch(Net *net, int step){
  for(auto* layer: net->layers()){
    if(layer->PreSyncF())
      MPI_Barrier(context_->mpicomm());
    layer->ComputeFeature();
    if(layer->PostSyncF())
      MPI_Barrier(context_->mpicomm());
  }
  return net->output_layer(0)->CalcPerf(true, true);
}

void Solver::TimeOneBatch(int runs) {
  phase=Phase::kTrain;
  Prefetcher prefetcher(train_shard_, net_);
  prefetcher();
  for(auto* layer:net_->input_layer())
    layer->SetInputData(nullptr);

  auto layers=net_->layers();
  int nlayers=layers.size();
  double* forward=new double[nlayers+1];;
  double* backward=new double[nlayers+1];;
  double* refresh=new double[nlayers+1];;
  double* sync=new double[nlayers+1];;
  memset(forward, 0, sizeof(double)*(1+nlayers));
  memset(backward, 0, sizeof(double)*(1+nlayers));
  memset(refresh, 0, sizeof(double)*(1+nlayers));
  memset(sync, 0, sizeof(double)*(1+nlayers));

  /*
  if(context_->group_id()!=0)
    sleep(10000);
    */
  LOG(ERROR)<<"Time One Batch...";;
  double sync_start, refresh_start, comp_start;
  //delegate_->StartCollectThread();
  for(auto* param: net_->params()){
      delegate_->AsyncGet(param,step_);
  }

  MPI_Barrier(context_->mpicomm());
  double start=Now();
  for(int k=0;k<runs;k++){
    int layerid=0;
    for(auto* layer: layers){
      refresh_start=Now();
      for(auto* param: layer->GetParams()){
          delegate_->AsyncCollect(param, step_);
      }
      sync_start=Now();
      refresh[layerid]+=sync_start-refresh_start;
      if(layer->PreSyncF())
        MPI_Barrier(context_->mpicomm());
      comp_start=Now();
      sync[layerid]+=comp_start-sync_start;
      layer->ComputeFeature();
      sync_start=Now();
      forward[layerid]+=sync_start-comp_start;
      if(layer->PostSyncF())
        MPI_Barrier(context_->mpicomm());
      sync[layerid]+=Now()-sync_start;
      layerid++;
    }

    for (auto layer = layers.rbegin(); layer != layers.rend(); layer++){
      layerid--;
      sync_start=Now();
      if((*layer)->PreSyncG())
        MPI_Barrier(context_->mpicomm());
      comp_start=Now();
      sync[layerid]+=comp_start-sync_start;
      (*layer)->ComputeGradient();
      refresh_start=Now();
      backward[layerid]+=refresh_start-comp_start;

      for(auto* param: (*layer)->GetParams()){
          delegate_->Update(param, step_);
          delegate_->AsyncGet(param, step_+1);
        }
      sync_start=Now();
      refresh[layerid]+=sync_start-refresh_start;
      if((*layer)->PostSyncG())
        MPI_Barrier(context_->mpicomm());
      sync[layerid]+=Now()-sync_start;
    }
    IncStep();
    if(GlobalContext::Get()->rank()==0)
      LOG(ERROR)<<"one iter";
  }
  //delegate_->StopCollectThread();
  double total=Now()-start;
  MPI_Barrier(context_->mpicomm());
  LOG(ERROR)<<"Finish";
  int K=1024;
  char buf[10*K];
  sprintf(buf, "\n");
  for(int i=0;i<nlayers;i++){
    sprintf(buf+strlen(buf), "Layer %10s forward %6.2f backward %6.2f sync %6.2f refresh %6.2f\n",
        layers[i]->name().c_str(),forward[i]/runs, backward[i]/runs, sync[i]/runs, refresh[i]/runs);
    forward[nlayers]+=forward[i];
    backward[nlayers]+=backward[i];
    sync[nlayers]+=sync[i];
    refresh[nlayers]+=refresh[i];
  }
  double armcitime=GAry::comm_time;
  sprintf(buf+strlen(buf), "Total\t%6.2f\tforward\t%6.2f\tbackward\t%6.2f\tcomp\t%6.2f\tsync\t%6.2f\trefresh\t%6.2f\tarmci\t%6.2f\n",
      total/runs,forward[nlayers]/runs, backward[nlayers]/runs, (forward[nlayers]+backward[nlayers]-armcitime)/runs, sync[nlayers]/runs,
      refresh[nlayers]/runs, armcitime/runs);
  LOG(ERROR)<<string(buf);
  delete forward;
  delete backward;
  delete sync;
  delete refresh;
  //DebugInfo(net_);
}

/***********************************************************************
 * Prefetcher Implementation
 ***********************************************************************/
Prefetcher::Prefetcher(string path, Net* _net) {
  net_=_net;
  shard_=new Shard(path, Shard::kRead);
}

Prefetcher::~Prefetcher() {
  delete shard_;
}

void Prefetcher::NextRecord(Record* record){
  string key;
  if(!shard_->Next(&key, record)){
    shard_->SeekToFirst();
    CHECK(shard_->Next(&key, record));
  }
}

void Prefetcher::operator()(){
  const DAry& input= net_->input_layer(0)->GetData(nullptr);
  Range nrng=input.IndexRange(0);
  Record record;
  for(int n=0;n<nrng.second-nrng.first;++n){
    NextRecord(&record);
    for(auto* layer:net_->input_layer())
      layer->AddInputRecord(record);
  }
}

}  // namespace lapis
