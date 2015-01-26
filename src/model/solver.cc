#include <glog/logging.h>
#include <mpi.h>
#include <random>
#include <vector>
#include "proto/model.pb.h"
#include "model/solver.h"
#include "core/common.h"
#include "utils/singleton.h"
#include "utils/factory.h"
#include "utils/timer.h"

namespace singa {
Solver::Solver(const SolverProto &proto) {
  // if step_>0, then the trainer is restored from a checkpoint
  step_ = proto.step();
  context_=GlobalContext::Get();
  proto_=proto;

  int ngroups=context_->num_groups();
  // update proto_, all groups will do the training, but only the first group
  // conduts test/validation. Hence, the training mini-batch is partitioned
  // onto all groups.
  if(context_->group_id()==0){
    if(proto_.batchsize()%ngroups!=0)
      LOG(ERROR)<<"batchsize % ngroups is not 0. "
      <<proto_.batchsize()<<" % "<<ngroups;
    proto_.set_validation_steps(proto_.validation_steps()*ngroups);
    proto_.set_validation_frequency(proto_.validation_frequency()*ngroups);
    proto_.set_test_steps(proto_.test_steps()*ngroups);
    proto_.set_test_frequency(proto_.test_frequency()*ngroups);
  }
  proto_.set_batchsize(proto_.batchsize()/ngroups);
  phase_=kTrain;
  string data_folder=GlobalContext::Get()->data_folder();
  train_shard_=data_folder+"/"+proto.train_folder();
  validation_shard_=data_folder+"/"+proto.validation_folder();
  test_shard_=data_folder+"/"+proto.test_folder();
  if(context_->num_servers()==0){ // will update parameters locally
    auto factory=Singleton<Factory<TableServerHandler>>::Instance();
    std::shared_ptr<TableServerHandler> tshandler(
        factory->Create(proto.sgd().handler()));
    tshandler->Setup(proto.sgd());
    delegate_=new TableDelegate(GlobalContext::Get(), tshandler);
  }else
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
  if(context_->group_id()==0)
    DLOG(ERROR)<<net->ToString();
  return net;
}

void Solver::PopulateTableServer(Net* net){
  for(auto* param: net->params()){
    param->Init();
    delegate_->Put(param);
  }
}

void Solver::ToProto(SolverProto *proto) {
  // TODO use the original proto (this proto_ is updated in constructor).
  proto->MergeFrom(proto_);
  proto->set_step(step_);
}

Performance Solver::Test(Net*net, const Phase& phase){
  string shard;
  int nsteps=0;
  if(phase==Phase::kValidation){
    shard=validation_shard_;
    nsteps=proto_.validation_steps();
  } else if(phase==Phase::kTest){
    shard=test_shard_;
    nsteps=proto_.test_steps();
  } else
    LOG(ERROR)<<"Phase must be kValidation or kTest";

  Prefetcher prefetcher(shard, net, phase, proto_.random_skip());
  std::thread *thd=nullptr;
  thd=new std::thread(std::ref(prefetcher));
  phase_=phase;
  Performance perf;
  for(int b=0;b<nsteps;b++){
    thd->join();
    delete thd;
    for(auto* layer:net->input_layer())
      layer->SetInputData(nullptr);
    if(b!=nsteps-1)
      thd=new std::thread(std::ref(prefetcher));
    perf.Aggregate(TestOneBatch(net, step_));
  }
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
  Performance trainperf;
  while (!HasFinished()) {
    phase_=kTrain;
    thd->join();
    delete thd;
    for(auto* layer:net->input_layer())
      layer->SetInputData(nullptr);
    if(!ValidateNow()&&!TestNow())
      thd=new std::thread(std::ref(prefetcher));
    trainperf.Aggregate(TrainOneBatch(net, step_));
    if(DisplayNow()&&context_->group_id()==0){
      ReportPerformance("Train", trainperf.Avg());
      DebugInfo(net);
      trainperf.Reset();
    }
    if(ValidateNow()&&context_->group_id()==0){
      Performance perf=Test(net, Phase::kValidation);
      ReportPerformance("Val  ", perf.Avg());
    }
    if(TestNow()&&context_->group_id()==0){
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
  if(context_->group_id()==0)
    return;
  char display[4096];
  auto layers=net->layers();
  DLOG(INFO)<<"Train Step: "<<step_;
  for(auto* layer: layers){
    sprintf(display, "Forward layer  %10s data norm1 %13.9f",
        layer->name().c_str(), layer->data().Norm1());
    DLOG(INFO)<<string(display);
  }
  for (auto layer = layers.rbegin(); layer != layers.rend(); layer++){
    if(!(*layer)->has_input()){
      sprintf(display, "Backward layer %10s grad norm1 %13.9f",
          (*layer)->name().c_str(), (*layer)->grad().Norm1());
      DLOG(INFO)<<string(display);
    }
  }
  for(auto* layer: layers){
    for(auto* param: layer->GetParams()){
      sprintf(display, "Layer %10s, param id %2d, name %10s, value norm1 %13.9f , grad norm1 %13.9f",
          layer->name().c_str(), param->id(),
          param->name().c_str(), param->data().Norm1(), param->grad().Norm1());
      DLOG(INFO)<<string(display);
    }
  }
}



Performance Solver::TrainOneBatch(Net *net, int step){
  //Timer tick;
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
  PerformanceLayer* perflayer=net->performance_layer(0);
  auto &srclayers=net->name2srclayers(perflayer->name());
  auto perf=perflayer->ComputePerformance(srclayers, kLoss|kPrecision);
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
  //LOG(ERROR)<<"Train one batch "<<tick.elapsed();
  return perf;
}

Performance Solver::TestOneBatch(Net *net, int step){
  //  Timer tick;
  for(auto* layer: net->layers()){
    const vector<Layer*> &srclayers=net->name2srclayers(layer->name());
    if(layer->PreSyncF(srclayers))
      MPI_Barrier(context_->mpicomm());
    layer->ComputeFeature(srclayers);
    if(layer->PostSyncF(srclayers))
      MPI_Barrier(context_->mpicomm());
  }
  PerformanceLayer* perflayer=net->performance_layer(0);
  const vector<Layer*> &srclayers=net->name2srclayers(perflayer->name());
  Performance perf=perflayer->ComputePerformance(srclayers,kPrecision);
  //LOG(ERROR)<<"Test one batch "<<tick.elapsed();
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

  LOG(ERROR)<<"Time One Batch...";;
  double sync_start, refresh_start, comp_start;
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
      if(layer->PreSyncF(srclayers)){
        MPI_Barrier(context_->mpicomm());
        CHECK(0)<<"presyncf "<<layer->name();
      }
      comp_start=Now();
      sync[layerid]+=comp_start-sync_start;
      layer->ComputeFeature(srclayers);
      sync_start=Now();
      forward[layerid]+=sync_start-comp_start;
      if(layer->PostSyncF(srclayers)){
        MPI_Barrier(context_->mpicomm());
        CHECK(0)<<"postsyncf "<<layer->name();
      }
      sync[layerid]+=Now()-sync_start;
      layerid++;
    }

    for (auto layer = layers.rbegin(); layer != layers.rend(); layer++){
      const vector<Layer*> &srclayers=net->name2srclayers((*layer)->name());
      layerid--;
      sync_start=Now();
      if((*layer)->PreSyncG(srclayers)){
        MPI_Barrier(context_->mpicomm());
        CHECK(0)<<"presyncg "<<(*layer)->name();
      }
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
      if((*layer)->PostSyncG(srclayers)){
        MPI_Barrier(context_->mpicomm());
        CHECK(0)<<"postsyncg "<<(*layer)->name();
      }
      sync[layerid]+=Now()-sync_start;
    }
    IncStep();
    //if(GlobalContext::Get()->group_id()==0)
      LOG(ERROR)<<"one iter"<<GlobalContext::Get()->rank();
  }
  MPI_Barrier(context_->allworkers_comm());
  double total=Now()-start;
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
Prefetcher::Prefetcher(string path, Net* net, Phase phase, bool random_skip) {
  net_=net;
  shard_=new shard::Shard(path, shard::Shard::kRead);
  phase_=phase;
  if(phase_==kTrain && random_skip){
    int nrecords=shard_->Count();
    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution(0,nrecords);
    int nskip=distribution(generator);
    LOG(INFO)<<"Random Skip "<<nskip<<" training records";
    for(int i=0;i<nskip;i++){
      Record record;
      NextRecord(&record);
    }
  }
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
  Pair nrng=input.localRange(0);
  Record record;
  //Timer tick;
  for(int n=0;n<nrng.second-nrng.first;++n){
    NextRecord(&record);
    for(auto* layer:net_->input_layer())
      layer->AddInputRecord(record, phase_);
  }
  //LOG(ERROR)<<"prefetch time "<<tick.elapsed();
}
}  // namespace singa
