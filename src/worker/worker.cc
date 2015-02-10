#include <glog/logging.h>
#include <memory>
#include "worker.h"
#include "proto/model.pb.h"
#include "utils/cluster.h"

namespace singa {
Worker::Worker(shared_ptr<Cluster> cluster){
  cluster_=cluster;
}

void Worker::Start(const ModelProto& model){
  modelproto_=model;
  LOG(ERROR)<<"Worker on "<<cluster_->hostname()<<" is starting...";
  train_net_=SetupNeuralNet(model.neuralnet(), kTrain);
  test_net_=SetupNeuralNet(model.neuralnet(), kTest);
  if(test_net_!=nullptr)
    test_net_.ShareWeights(train_net_);
  validation_net_=SetupNeuralNet(model.neuralnet(), kValidation);
  if(validation_net_!=nullptr)
    validation_net_.ShareWeights(train_net_);

  if(cluster_->num_servers()){ // training with parameter servers
    delegate_=make_shared<Delegate>(cluster_);
    if(cluster_->group_id()==0){ // first group populate the parameter servers
      train_net_->InitParams();
      delegate_->Put(train_net_->GetParams());
    }else{
      //delegate_->AsyncGet();
    }
  }else{ // training without parameter servers, update params locally
    if(model.updater().handler()==ParamUpdater_Type_kAdaGrad){
      updater_=make_shared<AdaGradUpdater>();
      updater_->Init(modelproto_.updater());
    } else
      LOG(FATAL)<<"Only support AdaGradUpdater now";
  }

  // update proto_, all groups will do the training, but only the first group
  // conduts test/validation. Hence, the training mini-batch is partitioned
  // onto all groups.
  if(cluster_->group_id()==0){
    int ngroups=cluster_.num_groups();
    modelproto_.set_validation_frequency(
        modelproto_.validation_frequency()/ngroups);
    modelproto_.set_test_frequency(modelproto_.test_frequency()/ngroups);
  }
  // todo, two ways to syn all workers
  // 1. create MPI communicator for all workers, and call MPI_Barrier for
  // this communicator
  // 2. handle_get returns false if the key of get() is not found in the
  // table, i.e., parameters have not been inserted
  MPI_Barrier(Cluster::Get()->worker_comm());
  Run(0);
  //solver.TimeOneBatch(net, 5);
  //LOG(ERROR)<<"Worker on "<<hostname_<< " is shutting down";
}

void Worker::Resume() {
  // TODO implement resume from snapshot
}


void PrefetchData(shared_ptr<NeuralNet> net){
  auto& layers=net->layers();
  CHECK(layers[0]->is_datalayer());
  for(auto& layer: layers){
    if(layer->is_datalayer())
      layer->ComputeFeature();
    else if(layer->is_parserlayer())
      layer->Prefetching();
  }
}

void Worker::Run(const int start_step){
  int step=start_step;
  Performance perf(train_net_);
  prefetch_thread_=std::thread(Worker::PrefetchData, train_net_);

  while(!StopNow(step){
    RunOneBatch(step, &perf);
    // communicate with others
    step++;
  }

  if(prefetch_thread_.joinable())
    prefetch_thread_.join();
}

void RunOneBatch(int step, Performance* perf){
  if(ValidateNow(step))
    Test(validation_net_);
  if(TestNow(step))
    Test(test_net_);

  TrainOneBatch(step);
  perf->Update();

  if(DisplayNow(step)){
    perf->Display();
    perf->Reset();
    DebugInfo(train_net_);
  }
}

shared_ptr<NeuralNet> SetupNeuralNet(const NeuralNetProto& np, Phase phase){
  // only consider training phase now.
  // resetting the proto to config test and valdiation neuralnet.
  shared_ptr<NeuralNet> net(new NeuralNet(np));
  return net;
}

void Test(shared_ptr<NeuralNet> net){
  std::thread prefetch;
  prefetch=std::thread(Worker::PrefetchData, net);
  Performance perf(net);
  for(int b=0;b<nsteps;b++){
    if(prefetch.joinable()){
      prefetch->join();
      for(auto* layer:net->parser_layers())
        layer->CompleteParsing();
      prefetch=std::thread(Worker::PrefetchData, net);
    }
    for(auto* layer: net->layers()){
      layer->ComputeFeature();
    }
    perf.Update();
  }
  if(prefetch.joinable()){
    prefetch.join();
  perf.Display();
}

void Solver::TimeOneBatch(shared_ptr<NeuralNet> net, int runs) {
  // prepare one batch training data
  Prefetching(net);
  for(auto& layer:net->parserlayers())
    layer->CompletePrefetching();

  // set up counters
  auto layers=net->layers();
  int nlayers=layers.size();
  vector<double> forward(nlayers+1, 0.0);
  vector<double> backward(nlayers+1, 0.0);
  vector<double> refresh(nlayers+1, 0.0);
  vector<double> sync(nlayers+1, 0.0);

  LOG(INFO)<<"Time One Batch...";;
  double sync_start, refresh_start, comp_start;

  if(updater_==nullptr)
  for(auto* param: net->GetParams()){
    delegate_->AsyncGet(param,step_);
  }

  CHECK_NE(cluster_->mycomm(), MPI_COMM_NULL);
  MPI_Barrier(cluster_->mycomm());
  double start=Now();
  for(int step=0;step<runs;step++){
    LOG(INFO)<<runid<<"-th run";
    int layerid=0;
    for(auto& layer: layers){
      refresh_start=Now();
      if(updater_==nullptr)
        for(auto* param: layer->GetParams()){
          delegate_->AsyncCollect(param, step);
        }
      sync_start=Now();
      refresh[layerid]+=sync_start-refresh_start;
      comp_start=Now();
      sync[layerid]+=comp_start-sync_start;
      layer->ComputeFeature();
      sync_start=Now();
      forward[layerid]+=sync_start-comp_start;
      sync[layerid]+=Now()-sync_start;
      layerid++;
    }

    for (auto layer = layers.rbegin(); layer != layers.rend(); layer++){
      layerid--;
      sync_start=Now();
      comp_start=Now();
      sync[layerid]+=comp_start-sync_start;
      (*layer)->ComputeGradient();
      refresh_start=Now();
      backward[layerid]+=refresh_start-comp_start;

      if(updater_)
        updater_->Update(param, step);
      else
        for(auto* param: (*layer)->GetParams()){
          delegate_->Update(param, step);
          delegate_->AsyncGet(param, step+1);
        }
      sync_start=Now();
      refresh[layerid]+=sync_start-refresh_start;
      sync[layerid]+=Now()-sync_start;
    }
  }
  MPI_Barrier(cluster_->worker_comm());
  double total=Now()-start;

  // display the time counters
  string disp="\n";
  char buf[1024];
  for(int i=0;i<nlayers;i++){
    sprintf(buf,
        "Layer %10s forward %6.2f backward %6.2f sync %6.2f refresh %6.2f\n",
        layers[i]->name().c_str(),forward[i]/runs, backward[i]/runs,
        sync[i]/runs, refresh[i]/runs);
    disp+=string(buf);
    forward[nlayers]+=forward[i];
    backward[nlayers]+=backward[i];
    sync[nlayers]+=sync[i];
    refresh[nlayers]+=refresh[i];
  }
  double armcitime=0.;//GAry::comm_time;
  sprintf(buf,
      "Total\t%6.2f\tforward\t%6.2f\tbackward\t%6.2f\tcomp\t%6.2f\tsync\t%6.2f\
      \trefresh\t%6.2f\tarmci\t%6.2f\n",
      total/runs,forward[nlayers]/runs, backward[nlayers]/runs,
      (forward[nlayers]+backward[nlayers]-armcitime)/runs, sync[nlayers]/runs,
      refresh[nlayers]/runs, armcitime/runs);
  disp+=string(buf);
  LOG(ERROR)<<disp;
}

void TrainOneBatch(int step){
  if(step==0){
    for(auto* param: train_net_->params()){
      delegate_->AsyncGet(param,step);
    }
  }
  if(prefetch_thread_.joinable()){
    prefetch_thread_->join();
    for(auto* layer:train_net_->parser_layers())
      layer->CompleteParsing();
    prefetch_thread_=std::thread(Worker::PrefetchData, train_net_);
  }
  if(updater_==nullptr){
    for(auto* param: train_net_->GetParams()){
      delegate_->AsyncGet(param,step_);
    }
  }
  auto& layers=train_net_->layers();
  for(auto& layer: layers){
    if(updater_==nullptr){
      for(auto* param: layer->GetParams()){
        delegate_->AsyncCollect(param, step);
      }
    }
    layer->ComputeFeature();
  }

  for (auto layer = layers.rbegin(); layer != layers.rend(); layer++){
    (*layer)->ComputeGradient();
    if(updater_==nullptr){
      for(auto* param: (*layer)->GetParams()){
        delegate_->Update(param, step);
        delegate_->AsyncGet(param, step+1);
      }
    }
    else{
      updater_->Update(param, step);
    }
  }
}

/*********************Implementation for Performance class*******************/
Performance::Performance(shared_ptr<NeuralNet> net){
  net_=net;
  auto& losslayers=net->losslayers();
  name_.resize(losslayers.size());
  metric_.resize(losslayers.size());
  for(auto& layer: net->losslayers()){
    name_.push_back(layer->name());
    metric_.push_back(vector<float>{});
    metric_.back().resize(layer->metric().count(),0.f);
  }
}

void Performance::Update(){
  const auto& losslayers=net->losslayers();
  for(size_t i=0;i<losslayers.size();i++){
    const float * ptr=losslayers[i]->metric().cpu_data();
    vector<float> m=metric_.at(i);
    for(int j=0;j<layers[i]->metric().count();j++)
      m[j]+=ptr[j];
  }
  counter_++;
}

void Performance::Reset(){
  for(auto& m: metric_)
    for(auto& x: m)
      x=0.f;
  }
}

void Performance::Display(){
  for(size_t i=0;i<metric_.size();i++){
    LOG(INFO)<<"Output from layer: "<<name_[i];
    string disp="";
    vector<float> m=metric_.at(i);
    for(size_t j=0;j<m.size();j++)
        disp+=std::to_string(j)<<" : "<<m[j]/counter_<<"\t";
    LOG(INFO)<<"\t"<<disp;
  }
}

}  // namespace singa
