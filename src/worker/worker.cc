#include <glog/logging.h>
#include <thread>
#include <memory>
#include <iostream>
#include "worker/worker.h"
#include "proto/model.pb.h"
#include "utils/cluster.h"
using std::thread;
namespace singa {
Worker::Worker(shared_ptr<Cluster> cluster){
  cluster_=cluster;
}

void Worker::Start(ModelProto model){
  LOG(ERROR)<<"Worker on "<<cluster_->hostname()<<" is starting...";
  // update proto_, all groups will do the training, but only the first group
  // conduts test/validation. Hence, the training mini-batch is partitioned
  // onto all groups.
  if(cluster_->groupid()==0){
    int ngroups=cluster_->ngroups();
    model.set_validation_frequency(
        model.validation_frequency()/ngroups);
    model.set_test_frequency(model.test_frequency()/ngroups);
  }
  train_net_=SetupNeuralNet(model.neuralnet(), model.prefetch(), kTrain);
  if(model.test_steps()){
    test_net_=SetupNeuralNet(model.neuralnet(), model.prefetch(), kTest);
    if(test_net_!=nullptr)
      test_net_->ShareWeights(train_net_);
  }
  if(model.validation_steps()){
    validation_net_=SetupNeuralNet(model.neuralnet(), model.prefetch(),
        kValidation);
    if(validation_net_!=nullptr)
      validation_net_->ShareWeights(train_net_);
  }

  pm_=make_shared<ParamManager>(train_net_, model.updater());
  pm_->InitParams(); //init local params

  Setup(0, model); //setup main executor
  int nthreads=cluster_->nthreads_per_procs();
  vector<Executor*> executors(nthreads-1);
  vector<thread> threads;
  for(size_t i=1;i<executors.size();i++){
    executors[i]=new Executor(i, model,  cluster_,pm_, train_net_);
    threads.push_back(thread(&Executor::Run, executors[i], 0));
  }

  // warmup to get computation speed
  Performance perf(train_net_);
  int64_t start=zclock_mono();
  for(int i=0;i<model.updater().warmup_steps();i++){
    RunOneBatch(i, &perf);
  }
  int64_t end=zclock_mono();
  pm_->SyncConfig((end-start)/1000.0f/model.updater().warmup_steps());

  if(cluster_->nservers()){
    if(cluster_->groupid()==0)
      pm_->SendParamsToServers();
    else
      pm_->GetParamsFromServers(model.updater().warmup_steps());
  }

  Run(model.updater().warmup_steps());
  for(auto& th: threads)
    th.join();
  for(size_t i=1;i<executors.size();i++){
    delete executors[i];
  }
}

void Worker::Resume() {
  // TODO implement resume from snapshot
}

shared_ptr<NeuralNet> Worker::SetupNeuralNet(const NetProto& np, bool prefetch,
    Phase phase){
  NetProto proto;
  proto.set_partition_type(np.partition_type());
  // exclude layers if necessary
  for(auto& layer:np.layer()){
    bool include=true;
    for(int x: layer.exclude()){
      if(x==phase)
        include=false;
    }
    if(include){
      LayerProto* lp=proto.add_layer();
      lp->CopyFrom(layer);
    }
  }
  LOG(INFO)<<"NeuralNet config is "<<proto.DebugString();
  shared_ptr<NeuralNet> net(new NeuralNet(proto));
  // set prefetch
  for(auto& layer: net->parserlayers()){
    layer->set_prefetch(prefetch);
  }
  for(auto& layer: net->datalayers()){
    layer->set_prefetch(prefetch);
  }
  return net;
}

// main working thread
void Worker::Run(const int start_step){
  step_=start_step;
  Performance perf(train_net_);
  while(!StopNow(step_)){
    RunOneBatch(step_, &perf);
    step_++;
    // TODO communicate with others, e.g., zookeeper
  }
}

/**************************Executor***********************************/
Executor::Executor(int local_threadid, const ModelProto& model,
    shared_ptr<Cluster> cluster,
    shared_ptr<ParamManager> pm,
    shared_ptr<NeuralNet> train_net,
    shared_ptr<NeuralNet> test_net,
    shared_ptr<NeuralNet> validation_net):
      cluster_(cluster),
      pm_(pm),
      train_net_(train_net),
      test_net_(test_net),
      validation_net_(validation_net){
        Setup(local_threadid, model);
      }

void Executor::Setup(int local_threadid, const ModelProto& model){
  tForward_=tBackward_=tSyncData_=tSyncParam_=0;
  modelproto_=model;
  local_threadid_=local_threadid;
  if(model.prefetch()){
    for(auto& layer: train_net_->datalayers()){
      if(cluster_->group_threadid(local_threadid_)==layer->locationid())
        localDataLayers_.push_back(layer);
    }
    if(localDataLayers_.size())
      prefetch_thread_=std::thread(Executor::PrefetchData,
          std::ref(localDataLayers_), true,1);
  }
  int gthreadid=cluster_->group_threadid(local_threadid);

  // for transfer data due to Model Partition
  for(auto& layer: train_net_->layers()){
    if(layer->locationid()==gthreadid){
      int pushloc=-1;
      if(layer->is_bridgesrclayer())
        pushloc=layer->dstlayers()[0]->locationid();
      else if(layer->is_bridgedstlayer())
        pushloc=layer->srclayers()[0]->locationid();
      if(pushloc!=-1&&push_.find(pushloc)==push_.end()){
        string endpoint="@tcp://*:"+cluster_->pull_port(local_threadid);
        pull_=zsock_new_pull(endpoint.c_str());
        string pushaddr=cluster_->group_thread_addr(pushloc);
        endpoint=">tcp://"+pushaddr
            +":"+cluster_->pull_port(pushloc%cluster_->nthreads_per_procs());
        push_[pushloc]= zsock_new_push(endpoint.c_str());
      }
    }
  }
}

Executor::~Executor(){
  if(prefetch_thread_.joinable())
    prefetch_thread_.join();
}

void Executor::PrefetchData(const vector<DataLayer*>& datalayers, bool training,
    int steps){
  if(datalayers.size()==0)
    return;
  for(int i=0;i<steps;i++){
    for(auto& layer: datalayers){
      layer->Prefetching(training);
      for(auto& dstlayer: layer->dstlayers()){
        CHECK(dstlayer->is_parserlayer());
        auto parserlayer=static_cast<ParserLayer*>(dstlayer.get());
        parserlayer->Prefetching(training);
      }
    }
  }
}

void Executor::Run(int step){
  step_=step;
  while(!StopNow(step_)){
    RunOneBatch(step_);
    step_++;
  }
}

void Executor::RunOneBatch(int step, Performance* perf){
  //DLOG(ERROR)<<"Step "<<step;
  ticks_++;
  // Test will call Pull which updates the sync time
  // Hence we store the sync time, and restore it later
  float tSyncData=tSyncData_, tSyncParam=tSyncParam_;
  if(ValidateNow(step)){
    LOG(ERROR)<<"Validation at step "<<step;
    Test(validation_net_, modelproto_.validation_steps(), perf!=nullptr);
  }
  if(TestNow(step)){
    LOG(ERROR)<<"Test at step "<<step;
    Test(test_net_, modelproto_.test_steps(), perf!=nullptr);
  }
  tSyncData_=tSyncData;
  tSyncParam_=tSyncParam;

  TrainOneBatch(step);
  if(perf!=nullptr){
    perf->Update();
    if(DisplayNow(step)){
      LOG(ERROR)<<"Training at step "<<step;
      LOG(ERROR)<<"\t"<<perf->ToString();
      perf->Reset();
      LOG(ERROR)<<"\t"<<TimerInfo();
    }
  }
}

void Executor::Pull(zsock_t* pull, shared_ptr<NeuralNet> net){
  int type;
  char *name;
  int64_t tick=zclock_mono();
  zframe_t* frame=zframe_new_empty();

  zsock_recv(pull_, "isf", &type, &name, &frame);
  if(type==kDataFrame){
    auto* dst=static_cast<BridgeDstLayer*>(
        net->name2layer(string(name)).get());
    memcpy(dst->mutable_data()->mutable_cpu_data(), zframe_data(frame),
        zframe_size(frame));
    dst->set_ready(true);
  }else if(type==kGradFrame){
    auto* src=static_cast<BridgeSrcLayer*>(net->name2layer(string(name)).get());
    memcpy(src->mutable_grad()->mutable_cpu_data(), zframe_data(frame),
        zframe_size(frame));
    src->set_ready(true);
  }
  zframe_destroy(&frame);
  delete name;
  tSyncData_+=zclock_mono()-tick;
}

void Executor::Forward(shared_ptr<NeuralNet> net, int step,  bool training){
  auto& layers=net->layers();
  for(auto& layer: layers){
    if(cluster_->group_procsid(layer->locationid())==cluster_->group_procsid()){
      if(layer->is_bridgedstlayer()){
        auto* dst=static_cast<BridgeDstLayer*>(layer.get());
        while(!dst->ready())
          Pull(pull_, train_net_);
      }
      if(training){
        for(shared_ptr<Param> p: layer->GetParams()){
          pm_->WaitUpdate(p, step, local_threadid_);
        }
      }
      layer->ComputeFeature(training);
      if(layer->is_bridgesrclayer()){
        zframe_t* frame=zframe_new(layer->data().cpu_data(),
            layer->data().count()*sizeof(float));
        zsock_send(push_[layer->locationid()], "isf",
            kDataFrame, layer->dstlayers()[0]->name().c_str(), frame);
        zframe_destroy(&frame);
      }
      if(training&&DisplayDebugInfo(step)&&layer->mutable_data()!=nullptr){
        LOG(INFO)<<StringPrintf("Forward layer  %10s data norm1 %13.9f",
            layer->name().c_str(), layer->data().asum_data());
      }
    }
  }
}

void Executor::Backward(shared_ptr<NeuralNet> net, int step){
  auto& layers=net->layers();
  for (auto it = layers.rbegin(); it != layers.rend(); it++){
    shared_ptr<Layer> layer=*it;
    if(cluster_->group_procsid(layer->locationid())==cluster_->group_procsid()){
      if(layer->is_bridgesrclayer()){
        auto* src=static_cast<BridgeSrcLayer*>(layer.get());
        while(!src->ready())
          Pull(pull_, train_net_);
      }
      layer->ComputeGradient();
      if(DisplayDebugInfo(step)&&layer->mutable_grad()!=nullptr){
        LOG(INFO)<<StringPrintf("Backward layer %10s grad norm1 %13.9f\t",
            layer->name().c_str(), layer->grad().asum_data());
        for(shared_ptr<Param> p: layer->GetParams())
          LOG(INFO)<<StringPrintf("param id %2d, name %10s,\
              value norm1 %13.9f, grad norm1 %13.9f",
              p->id(), p->name().c_str(),
              p->data().asum_data(), p->grad().asum_data());
      }
      for(shared_ptr<Param> p: layer->GetParams()){
        pm_->UpdateParam(p, step, local_threadid_);
      }
      if(layer->is_bridgedstlayer()){
        zframe_t* frame=zframe_new(layer->grad().cpu_data(),
            layer->data().count()*sizeof(float));
        zsock_send(push_[layer->locationid()], "isf",
            kGradFrame, layer->srclayers()[0]->name().c_str(), frame);
        zframe_destroy(&frame);
      }
    }
  }
}

void Executor::TrainOneBatch(int step){
  int64_t tick=zclock_mono();
  if(prefetch_thread_.joinable()){
      prefetch_thread_.join();
      prefetch_thread_=std::thread(Executor::PrefetchData,
          std::ref(localDataLayers_), true,1);
  }
  Forward(train_net_, step, true);
  tForward_+=zclock_mono()-tick;
  tick=zclock_mono();
  Backward(train_net_, step);
  tBackward_+=zclock_mono()-tick;
}

void Executor::Test(shared_ptr<NeuralNet> net, int nsteps, bool disperf){
  std::thread prefetch;
  vector<DataLayer*> localDataLayers;
  if(modelproto_.prefetch()){
    auto cluster=Cluster::Get();
    for(auto& layer: net->datalayers()){
      int locid=layer->locationid();
      if(cluster->group_threadid(local_threadid_)==locid)
        localDataLayers.push_back(layer);
    }
    if(localDataLayers.size())
      prefetch=std::thread(Executor::PrefetchData,  std::ref(localDataLayers),
          false,1);
  }
  Performance perf(net);
  for(int b=0;b<nsteps;b++){
    if(prefetch.joinable()){
      prefetch.join();
      if(b<nsteps-1)
        prefetch=std::thread(Executor::PrefetchData, std::ref(localDataLayers),
          false,1);
    }
    Forward(net, b, false);
    if(disperf)
      perf.Update();
  }
  if(prefetch.joinable())
    prefetch.join();
  if(disperf)
    LOG(ERROR)<<"\t"<<perf.ToString();
}
/*********************Implementation for Performance class*******************/
Performance::Performance(shared_ptr<NeuralNet> net):net_(net), counter_(0){
  for(auto& layer: net->losslayers()){
    name_.push_back(layer->name());
    metric_.push_back(vector<float>{});
    metric_.back().resize(layer->metric().count(),0.f);
  }
}

void Performance::Update(){
  const auto& losslayers=net_->losslayers();
  for(size_t i=0;i<losslayers.size();i++){
    const float * ptr=losslayers[i]->metric().cpu_data();
    vector<float>& m=metric_.at(i);
    for(int j=0;j<losslayers[i]->metric().count();j++)
      m[j]+=ptr[j];
  }
  counter_++;
}

void Performance::Reset(){
  for(auto& m: metric_)
    for(auto& x: m)
      x=0.f;
  counter_=0;
}

string Performance::ToString(){
  string disp="";
  for(size_t i=0;i<metric_.size();i++){
    disp+="Output from "+name_[i]+" layer ";
    vector<float> m=metric_.at(i);
    for(size_t j=0;j<m.size();j++)
        disp+=std::to_string(j)+" : "+std::to_string(m[j]/counter_)+"\t";
    disp+="\n";
  }
  return disp;
}

}  // namespace singa
