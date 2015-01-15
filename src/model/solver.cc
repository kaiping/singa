#include <glog/logging.h>
#include <mpi.h>
#include <vector>
#include "proto/model.pb.h"
#include "model/solver.h"
#include "utils/timer.h"

namespace singa {
Solver::Solver(const SolverProto &proto) {
  //! if step_>0, then the trainer is restored from a checkpoint
  step_ = proto.step();
  proto_=proto;
  context_=GlobalContext::Get();
  phase_=kTrain;

  string data_folder=GlobalContext::Get()->data_folder();
  train_shard_=data_folder+"/"+proto.train_folder();
  val_shard_=data_folder+"/"+proto.validation_folder();
  test_shard_=data_folder+"/"+proto.test_folder();
  delegate_=new TableDelegate(GlobalContext::Get());
}

Solver::~Solver() {
  delete delegate_;
}

Net* Solver::SetupNeuralNet(const NetProto& proto) {
  Net *net=new Net(proto);
  shard::Shard shard(train_shard_, shard::Shard::kRead);
  Record record;
  string key;
  shard.Next(&key, &record);
  // setup the net
  net->Setup(proto_.batchsize(), record);
  return net;
}

void Solver::PopulateTableServer(Net* net){
  for(auto* param: net->params()){
    param->Init();
    delegate_->Put(param);
  }
}

void Solver::ToProto(SolverProto *proto) {
  proto->MergeFrom(proto_);
  proto->set_step(step_);
}



Performance Solver::Test(Net*net, const Phase& phase){
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

  Prefetcher prefetcher(shard, net, phase);
  std::thread *thd=nullptr;
  thd=new std::thread(std::ref(prefetcher));
  phase_=phase;
  Performance perf;
  for(int b=0;b<nsteps;b++){
    thd->join();
    delete thd;
    for(auto* layer:net->input_layer())
      layer->SetInputData(nullptr);
    thd=new std::thread(std::ref(prefetcher));
    perf.Aggregate(TestOneBatch(net, step_));
  }
  thd->join();
  delete thd;
  phase_=Phase::kTrain;
  return perf;
}

void Solver::ReportPerformance(string prefix, Performance perf) {
  LOG(ERROR)<<"Train Step: "<<step_<<" "<<prefix<<" "<<perf.ToString();
}

void Solver::Train(Net* net, int start_step){
  step_=start_step;
  Prefetcher prefetcher(train_shard_, net, kTrain);
  std::thread *thd=nullptr;
  thd=new std::thread(std::ref(prefetcher));
  while (!HasFinished()) {
    phase_=kTrain;
    thd->join();
    delete thd;
    for(auto* layer:net->input_layer())
      layer->SetInputData(nullptr);
    if(!ValidateNow()&&!TestNow())
      thd=new std::thread(std::ref(prefetcher));
    train_perf_.Aggregate(TrainOneBatch(net, step_));
    if(DisplayNow()){
      ReportPerformance("Train", train_perf_.Avg());
      DebugInfo(net);
      train_perf_.Reset();
    }
    if(ValidateNow()){
      Performance perf=Test(net, Phase::kValidation);
      ReportPerformance("Val  ", perf.Avg());
    }
    if(TestNow()){
      Performance perf=Test(net, Phase::kTest);
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
      DArryProtoProto data;
      param->data().ToProto(&data, true);
      DArryProtoProto grad;
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
    const vector<Layer*> &srclayers=net->name2srclayers(layer->name());
    for(auto* param: layer->GetParams()){
        delegate_->AsyncCollect(param, step);
    }
    if(layer->PreSyncF(srclayers))
      MPI_Barrier(context_->mpicomm());
    layer->ComputeFeature(srclayers);
    if(layer->PostSyncF(srclayers))
      MPI_Barrier(context_->mpicomm());
  }
  const vector<Layer*> &srclayers=net->name2srclayers(net->performance_layer(0)->name());
  Performance perf=net->performance_layer(0)->ComputePerformance(srclayers, kLoss);
  for (auto layer = layers.rbegin(); layer != layers.rend(); layer++){
    const vector<Layer*> &srclayers=net->name2srclayers((*layer)->name());
    if((*layer)->PreSyncG(srclayers))
      MPI_Barrier(context_->mpicomm());
    (*layer)->ComputeGradient(srclayers);
    for(auto* param: (*layer)->GetParams()){
        delegate_->Update(param, step);
        delegate_->AsyncGet(param,step+1);
    }
    if((*layer)->PostSyncG(srclayers))
      MPI_Barrier(context_->mpicomm());
  }
  //DebugInfo(net);
  return perf;
}

Performance Solver::TestOneBatch(Net *net, int step){
  for(auto* layer: net->layers()){
    const vector<Layer*> &srclayers=net->name2srclayers(layer->name());
    if(layer->PreSyncF(srclayers))
      MPI_Barrier(context_->mpicomm());
    layer->ComputeFeature(srclayers);
    if(layer->PostSyncF(srclayers))
      MPI_Barrier(context_->mpicomm());
  }
  const vector<Layer*> &srclayers=net->name2srclayers(net->performance_layer(0)->name());
  Performance perf=net->performance_layer(0)->ComputePerformance(srclayers,kAccuracy);

  return perf;
}

void Solver::TimeOneBatch(Net* net, int runs) {
  phase_=Phase::kTrain;
  Prefetcher prefetcher(train_shard_, net, phase_);
  prefetcher();
  for(auto* layer:net->input_layer())
    layer->SetInputData(nullptr);

  auto layers=net->layers();
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
  for(auto* param: net->params()){
      delegate_->AsyncGet(param,step_);
  }

  MPI_Barrier(context_->mpicomm());
  double start=Now();
  for(int k=0;k<runs;k++){
    int layerid=0;
    for(auto* layer: layers){
      const vector<Layer*> &srclayers=net->name2srclayers(layer->name());
      refresh_start=Now();
      for(auto* param: layer->GetParams()){
          delegate_->AsyncCollect(param, step_);
      }
      sync_start=Now();
      refresh[layerid]+=sync_start-refresh_start;
      if(layer->PreSyncF(srclayers))
        MPI_Barrier(context_->mpicomm());
      comp_start=Now();
      sync[layerid]+=comp_start-sync_start;
      layer->ComputeFeature(srclayers);
      sync_start=Now();
      forward[layerid]+=sync_start-comp_start;
      if(layer->PostSyncF(srclayers))
        MPI_Barrier(context_->mpicomm());
      sync[layerid]+=Now()-sync_start;
      layerid++;
    }

    for (auto layer = layers.rbegin(); layer != layers.rend(); layer++){
      const vector<Layer*> &srclayers=net->name2srclayers((*layer)->name());
      layerid--;
      sync_start=Now();
      if((*layer)->PreSyncG(srclayers))
        MPI_Barrier(context_->mpicomm());
      comp_start=Now();
      sync[layerid]+=comp_start-sync_start;
      (*layer)->ComputeGradient(srclayers);
      refresh_start=Now();
      backward[layerid]+=refresh_start-comp_start;

      for(auto* param: (*layer)->GetParams()){
          delegate_->Update(param, step_);
          delegate_->AsyncGet(param, step_+1);
        }
      sync_start=Now();
      refresh[layerid]+=sync_start-refresh_start;
      if((*layer)->PostSyncG(srclayers))
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
  double armcitime=0.;//GAry::comm_time;
  sprintf(buf+strlen(buf), "Total\t%6.2f\tforward\t%6.2f\tbackward\t%6.2f\tcomp\t%6.2f\tsync\t%6.2f\trefresh\t%6.2f\tarmci\t%6.2f\n",
      total/runs,forward[nlayers]/runs, backward[nlayers]/runs, (forward[nlayers]+backward[nlayers]-armcitime)/runs, sync[nlayers]/runs,
      refresh[nlayers]/runs, armcitime/runs);
  LOG(ERROR)<<string(buf);
  delete forward;
  delete backward;
  delete sync;
  delete refresh;
  //DebugInfo(net);
}

/***********************************************************************
 * Prefetcher Implementation
 ***********************************************************************/
Prefetcher::Prefetcher(string path, Net* net, Phase phase) {
  net_=net;
  shard_=new shard::Shard(path, shard::Shard::kRead);
  phase_=phase;
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
  // can avoid directly dependent on DArryProto by fetching the whole mini-batch
  // or telling the prefetcher the size of partition of the mini-batch
  const DArray& input= net_->input_layer(0)->data();
  // add a lshape(k) api for DArryProto to return local shape on k-dim
  Pair nrng=input.LocalRange(0);
  Record record;
  for(int n=0;n<nrng.second-nrng.first;++n){
    NextRecord(&record);
    for(auto* layer:net_->input_layer())
      layer->AddInputRecord(record, phase_);
  }
}
}  // namespace singa
