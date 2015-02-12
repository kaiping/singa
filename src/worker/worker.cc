#include <glog/logging.h>
#include <thread>
#include <memory>
#include "worker.h"
#include "proto/model.pb.h"
#include "utils/cluster.h"
using std::thread;
namespace singa {
Worker::Worker(shared_ptr<Cluster> cluster){
  cluster_=cluster;
}

void Worker::Start(const ModelProto& model){
  LOG(ERROR)<<"Worker on "<<cluster_->hostname()<<" is starting...";
  train_net_=SetupNeuralNet(model.neuralnet(), kTrain);
  test_net_=SetupNeuralNet(model.neuralnet(), kTest);
  if(test_net_!=nullptr)
    test_net_.ShareWeights(train_net_);
  validation_net_=SetupNeuralNet(model.neuralnet(), kValidation);
  if(validation_net_!=nullptr)
    validation_net_.ShareWeights(train_net_);

  // update proto_, all groups will do the training, but only the first group
  // conduts test/validation. Hence, the training mini-batch is partitioned
  // onto all groups.
  if(cluster_->group_id()==0){
    int ngroups=cluster_.ngroups();
    model.set_validation_frequency(
        model.validation_frequency()/ngroups);
    model.set_test_frequency(model.test_frequency()/ngroups);
  }
  // todo, two ways to syn all workers
  // 1. create MPI communicator for all workers, and call MPI_Barrier for
  // this communicator
  // 2. handle_get returns false if the key of get() is not found in the
  // table, i.e., parameters have not been inserted
  ParamManager pm(train_net_, model.updater());
  if(cluster_->groupID()==0){
    pm.InitParams(); //init local params
  }else{
    pm.GetParamsFromServers(); // will be blocked until recv all parameters.
  }

  int nthreads=cluster_.nthreads_per_procs();
  vector<Executor> executors(nthreads-1);
  vector<thread> threads;
  Setup(model,0); //setup main executor
  for(size_t i=1;i<executors.size();i++){
    executors[i]=new Executor(i, model, cluster_, train_net_);
    threads.push_back(thread(&Executor::Run, ref(*executors[i]), &pm, 0));
  }
  if(cluster_->groupID()==0){
    Performance perf(train_net_);
    for(int i=0;i<model.warmup_steps();i++){
      RunOneBatch(i, &perf);
      pm.Update(i, train_net_, local_threadID_);
    }
    pm.SendParamsToServers();
  }
  Run(&pm, model.warmup_steps());
  for(auto& th: threads)
    th.join();
  for(size_t i=1;i<executors.size();i++){
    delete executors[i];
  }
  //LOG(ERROR)<<"Worker on "<<hostname_<< " is shutting down";
}

void Worker::Resume() {
  // TODO implement resume from snapshot
}

shared_ptr<NeuralNet> Worker::SetupNeuralNet(const NeuralNetProto& np,
    Phase phase){
  // only consider training phase now.
  // TODO reset the proto to config test and valdiation neuralnet.
  // diff nets should have diff layer objects,
  // but share parameter objects (by ptr).
  shared_ptr<NeuralNet> net(new NeuralNet(np));
  return net;
}

void Worker::Run(ParamManager* pm, const int start_step){
  step_=start_step;
  while(!StopNow(step_)){
    RunOneBatch(step_, &perf);
    pm->Update(step_, train_net_, local_threadID_);
    // communicate with others
    step_++;
  }
}


/**************************Executor***********************************/
void Executor::Executor(int local_threadID,
    shared_ptr<Cluster> cluster,
    shared_ptr<NeuralNet> train_net,
    shared_ptr<NeuralNet> test_net,
    shared_ptr<NeuralNet> validation_net):
      cluster_(cluster),
      train_net_(train_net),
      test_net_(test_net),
      validation_net_(validation_net){
        Setup(local_threadID);
      }

void Executor::Setup(int local_threadID, const ModelProto& model){
  modelproto_=model;
  local_threadID_=local_threadID;
  gthreadID=cluster_->group_threadID(local_threadID);
  if(train_net_->datalayer()->locationID()==gthreadID)
    prefetch_thread_=std::thread(Worker::PrefetchData, train_net_, true);

  subparam_=zsock_new_sub(">inproc://pmpub");
  pushparam_=zsock_new_push(">inproc://pmpull");

  for(auto& layer: train_net_->layers()){ //TODO check with PS
    if(layer->locationID()==gthreadID){
      int pushloc=-1;
      if(layer->is_bridgesrclayer())
        pushloc=layer->dstlayers()[0]->locationID();
      else if(layer->is_bridgedstlayer())
        pushloc=layer->srclayers()[0]->locationID();
      if(pushloc!=-1&&push_.find(pushloc)==push_.end()){
        pull_=zsock_new_pull("@tcp://*:"+cluster_->pull_port(local_threadID));
        string pushaddr=cluster_->addr(cluster_->global_procsID(pushloc));
        push_[pushloc]= zsock_new_push( ">tcp://"+pushaddr
            +":"+cluster_->pull_port(pushloc%cluster_->nthreads_per_procs()));
      }
    }
  }
}

void Executor::~Executor(){
  if(prefetch_thread_.joinable())
    prefetch_thread_.join();
}

void Executor::PrefetchData(shared_ptr<NeuralNet> net, bool training){
  auto& layers=net->layers();
  CHECK(layers[0]->is_datalayer());
  for(auto& layer: layers){
    if(layer->is_datalayer())
      layer->ComputeFeature(training);
    else if(layer->is_parserlayer())
      layer->Prefetching(training);
  }
}

void Executor::Run(ParamManager*pm, int step){
  step_=step;
  while(!StopNow(step_)){
    RunOneBatch(step_);
    pm->Update(train_net_, step_, local_threadID_);
    step_++;
  }
}

void Executor::RunOneBatch(int step, Performance* perf){
  ticks_++;
  // Test will call Pull which updates the sync time
  // Hence we store the sync time, and restore it later
  int64_t tSyncData=tSyncData_, tSyncParam=tSyncParam_;
  if(ValidateNow(step))
    Test(validation_net_, perf==nullptr);
  if(TestNow(step))
    Test(test_net_, perf==nullptr);
  tSyncData_=tSyncData;
  tSyncParam_=tSyncParam;

  TrainOneBatch(step);
  if(perf!=nullptr){
    perf->Update();
    if(DisplayNow(step)){
      LOG(INFO)<<perf->ToString();
      perf->Reset();
      LOG(INFO)<<TimerInfo();
      DLOG(INFO)<<DebugInfo(train_net_);
    }
  }
}

void Executor::Pull(zsock_t pull, shared_ptr<NeuralNet> net){
  int type;
  char *name;
  int64_t tick=zclock_mono();
  zframe_t* frame=zframe_new_empty();

  zsock_recv(pull_, "isf", &type, &name, &frame);
  if(type==kDataFrame){
    auto* dst=static_cas<BridgeDstLayer*>(
        net->name2layer(string(name).get()));
    dst->set_ready(true);
    memcpy(layer->mutable_data()->mutable_cpu_data(), zframe_data(frame),
        zframe_size(frame));
  }else if(type==kGradFrame){
    auto* src=static_cas<BridgeSrcLayer*>(net->name2layer(string(name).get()));
    src->set_ready(true);
    memcpy(layer->mutable_grad()->mutable_cpu_data(), zframe_data(frame),
        zframe_size(frame));
  }
  zframe_destroy(&frame);
  delete name;
  tSyncData_+=zclock_mono()-tick;
}

void Executor::Forward(shared_ptr<NeuralNet> net, bool training){
  auto& layers=net->layers();
  int type;
  int paramID;
  for(auto& layer: layers){
    while(!layer->ready()){
      Pull(pull_, train_net_);
    }
    layer->ComputeFeature(training);
    if(layer->is_bridgesrclayer()){
      zframe_t* frame=zframe_new(layer->data().cpu_data(),
          layer->data().count()*sizeof(float));
      zsock_send(push_[layer->locationID()], "isf",
          kDataFrame, layer->dstlayers()[0]->name().c_str(), frame);
      zframe_destroy(&frame);
    }
  }
}

void Executor::Backward(shared_ptr<NeuralNet> net){
  auto& layers=net->layers();
  for (auto layer = layers.rbegin(); layer != layers.rend(); layer++){
    while(!layer->ready()){
      Pull(pull_, train_net_);
    }
    (*layer)->ComputeGradient();
    if(layer->is_bridgedstlayer()){
      zframe_t* frame=zframe_new(layer->grad().cpu_data(),
          layer->data().count()*sizeof(float));
      zsock_send(push_[layer->locationID()], "isf",
          kGradFrame, layer->srclayers()[0]->name().c_str(), frame);
      zframe_destroy(&frame);
    }
  }
}

void Executor::TrainOneBatch(int step){
  if(prefetch_thread_.joinable()){
    prefetch_thread_->join();
    for(auto* layer:train_net_->parser_layers())
      layer->CompleteParsing();
    prefetch_thread_=std::thread(PrefetchData, train_net_, true);
  }
  int64_t tick=zclock_mono();
  Forward(train_net_, true);
  tForward_+=zclock_mono()-tick;
  tick=zclock_mono();
  Backward(train_net_);
  tBackward_+=zclock_mono()-tick;
}

void Executor::Test(shared_ptr<NeuralNet> net, bool disperf){
  std::thread prefetch;
  prefetch=std::thread(Worker::PrefetchData, net, false);
  Performance perf(net);
  for(int b=0;b<nsteps;b++){
    if(prefetch.joinable()){
      prefetch->join();
      for(auto* layer:net->parser_layers())
        layer->CompleteParsing();
      prefetch=std::thread(Worker::PrefetchData, net, false);
    }
    Forward(net, false);
    if(disperf)
      perf.Update();
  }
  if(prefetch.joinable())
    prefetch.join();
  if(disperf)
    LOG(INFO)<<perf.ToString();
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

string Performance::ToString(){
  string disp="";
  for(size_t i=0;i<metric_.size();i++){
    disp+="Output from layer: "+name_[i];
    vector<float> m=metric_.at(i);
    for(size_t j=0;j<m.size();j++)
        disp+=std::to_string(j)+" : "+std::to_string(m[j]/counter_)+"\t";
    disp+="\n";
  }
  return disp;
}

}  // namespace singa
