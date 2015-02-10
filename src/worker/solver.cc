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
}

Solver::~Solver() {
  delete delegate_;
}
/*
shared_ptr<Net> Solver::SetupNeuralNet(const NetProto& proto) {
  shared_ptr<Net> net(proto);
  if(cluster_->group_id()==0)
    DLOG(ERROR)<<net->ToString();
  return net;
}
*/

/*
void Solver::ToProto(SolverProto *proto) {
  // TODO use the original proto (this proto_ is updated in constructor).
  proto->MergeFrom(proto_);
  proto->set_step(step_);
}
*/

void Solver::Test(shared_ptr<Net>net, const Phase& phase){
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
  vector<shared_ptr<LossLayer>> losslayers;
  Performance perf(net->losslayers());
  for(int b=0;b<nsteps;b++){
    thd->join();
    delete thd;
    for(auto* layer:net->input_layer())
      layer->SetInputData(nullptr);
    if(b!=nsteps-1)
      thd=new std::thread(std::ref(prefetcher));
    TestOneBatch(net, step_);
    perf.Aggregate(net->losslayers());
  }
  perf.Display(net->losslayers());
 phase_=Phase::kTrain;
}


void Performance::Display(const vector<shared_ptr<LossLayer>>& layers){
  for(size_t i=0;i<layers.size();i++){
    LOG(INFO)<<"Output from layer: "<<losslayers[i]->name();
    string disp="";
    vector<float> m=metric.at(i);
    for(int j=0;j<layers[i]->output().count();j++)
        disp+=std::to_string(j)<<" : "<<m/counter_<<"\t";
    LOG(INFO)<<"\t"<<disp;
  }
}
void Solver::Train(shared_ptr<Net> net, int start_step){
  step_=start_step;
  Prefetcher prefetcher(train_shard_, net, kTrain);
  std::thread *thd=nullptr;
  thd=new std::thread(std::ref(prefetcher));
  Performance perf(net->losslayers());
  while (!HasFinished()) {
    phase_=kTrain;
    thd->join();
    delete thd;
    for(auto* layer:net->inputlayers())
      layer->CompletePrefetch();
    if(!ValidateNow()&&!TestNow())
      thd=new std::thread(std::ref(prefetcher));
    TrainOneBatch(net, step_);
    perf.Aggregate(net->losslayers());
    if(DisplayNow()&&cluster_->group_id()==0){
      perf.Display(net->losslayers());
      perf.Reset();
      DebugInfo(net);
    }
    if(ValidateNow()&&cluster_->group_id()==0){
      Test(net, Phase::kValidation);
    }
    if(TestNow()&&cluster_->group_id()==0){
      Test(net, Phase::kTest);
    }
    if(ValidateNow()||TestNow())
      thd=new std::thread(std::ref(prefetcher));
    IncStep();
  }
  thd->join();
  delete thd;
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
    layer->ComputeFeature(srclayers);
  }
  for (auto layer = layers.rbegin(); layer != layers.rend(); layer++){
    const vector<Layer*> &srclayers=net->name2srclayers((*layer)->name());
    (*layer)->ComputeGradient(srclayers);
    for(auto* param: (*layer)->GetParams()){
        delegate_->Update(param, step);
        delegate_->AsyncGet(param,step+1);
    }
  }
  //LOG(ERROR)<<"Train one batch "<<tick.elapsed();
  return perf;
}

Performance Solver::TestOneBatch(Net *net, int step){
  //  Timer tick;
  for(auto* layer: net->layers()){
    const vector<Layer*> &srclayers=net->name2srclayers(layer->name());
    layer->ComputeFeature(srclayers);
  }
}

void Solver::TimeOneBatch(Net* net, int runs) {
  phase_=Phase::kTrain;
  // prepare one batch training data
  Prefetcher prefetcher(train_shard_, net, phase_);
  prefetcher();
  for(auto* layer:net->input_layer())
    layer->SetInputData(nullptr);

  // set up counters
  auto layers=net->layers();
  int nlayers=layers.size();
  vector<double> forward(nlayers+1, 0.0);
  vector<double> backward(nlayers+1, 0.0);
  vector<double> refresh(nlayers+1, 0.0);
  vector<double> sync(nlayers+1, 0.0);

  LOG(INFO)<<"Time One Batch...";;
  double sync_start, refresh_start, comp_start;

  for(auto* param: net->params()){
    delegate_->AsyncGet(param,step_);
  }

  CHECK_NE(cluster_->mycomm(), MPI_COMM_NULL);
  MPI_Barrier(cluster_->mycomm());
  double start=Now();
  for(int runid=0;runid<runs;runid++){
    LOG(INFO)<<runid<<"-th run";
    int layerid=0;
    for(auto* layer: layers){
      const vector<Layer*> &srclayers=net->name2srclayers(layer->name());
      refresh_start=Now();
      for(auto* param: layer->GetParams()){
        delegate_->AsyncCollect(param, step_);
      }
      sync_start=Now();
      refresh[layerid]+=sync_start-refresh_start;
      comp_start=Now();
      sync[layerid]+=comp_start-sync_start;
      layer->ComputeFeature(srclayers);
      sync_start=Now();
      forward[layerid]+=sync_start-comp_start;
      sync[layerid]+=Now()-sync_start;
      layerid++;
    }

    for (auto layer = layers.rbegin(); layer != layers.rend(); layer++){
      const vector<Layer*> &srclayers=net->name2srclayers((*layer)->name());
      layerid--;
      sync_start=Now();
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
      sync[layerid]+=Now()-sync_start;
    }
    IncStep();
  }
  MPI_Barrier(cluster_->worker_comm());
  double total=Now()-start;

  // display the time counters
  int K=1024;
  char buf[10*K];
  sprintf(buf, "\n");
  for(int i=0;i<nlayers;i++){
    sprintf(buf+strlen(buf),
        "Layer %10s forward %6.2f backward %6.2f sync %6.2f refresh %6.2f\n",
        layers[i]->name().c_str(),forward[i]/runs, backward[i]/runs,
        sync[i]/runs, refresh[i]/runs);
    forward[nlayers]+=forward[i];
    backward[nlayers]+=backward[i];
    sync[nlayers]+=sync[i];
    refresh[nlayers]+=refresh[i];
  }
  double armcitime=0.;//GAry::comm_time;
  sprintf(buf+strlen(buf),
      "Total\t%6.2f\tforward\t%6.2f\tbackward\t%6.2f\tcomp\t%6.2f\tsync\t%6.2f\
      \trefresh\t%6.2f\tarmci\t%6.2f\n",
      total/runs,forward[nlayers]/runs, backward[nlayers]/runs,
      (forward[nlayers]+backward[nlayers]-armcitime)/runs, sync[nlayers]/runs,
      refresh[nlayers]/runs, armcitime/runs);
  LOG(INFO)<<string(buf);
}

}  // namespace singa
