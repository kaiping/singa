#include "utils/cluster.h"
#include "worker/param_manager.h"

namespace singa{

ParamManager::ParamManager(shared_ptr<NeuralNet> net,
    const UpdaterProto& updater){
  step_=-1;
  net_=net;
  hogwild_=updater.hogwild();
  if(updater.type()==UpdaterProto_Type_kAdaGrad){
    updater_=make_shared<AdaGradUpdater>();
    updater_->Init(updater);
  } else
    LOG(FATAL)<<"Only support AdaGradUpdater now";

  int count=0;
  shared_ptr<Cluster> cluster=Cluster::Get();
  for(shared_ptr<Layer> layer: net->layers()){
    if(cluster->group_procsid(layer->locationid())==cluster->group_procsid()){
      for(Param* p: layer->GetParams()){
        int ownerid=p->owner()->id();
        if(paramid2Offset_.find(ownerid)==paramid2Offset_.end()){
          paramid2Offset_[p->id()]=count;
          paramid2Offset_[ownerid]=count;
          count+=p->data().count();
        }else{
          paramid2Offset_[p->id()]=paramid2Offset_[ownerid];
        }
        ownerid2Params_[ownerid].push_back(p);
        //paramid2param_[p->id()]=p;
      }
    }
  }
  ParamProto pp;
  param_.Setup(pp, vector<int> {count});
  float* dptr=param_.mutable_cpu_data();
  for(auto& entry: ownerid2Params_){
    entry.second.at(0)->data().data()->set_cpu_data(
        dptr+paramid2Offset_[entry.first]);
  }
  if(cluster->nservers()>0){ // sync with parameter server
    string ps= cluster->server_addr(cluster->group_procsid());
    string endpoint=">tcp://"+ps+":"+cluster->pub_port();
    sub_=zsock_new_sub(endpoint.c_str(),"");
    CHECK(sub_!=nullptr);
    endpoint=">tcp://"+ps+":"+cluster->pull_port();
    push_=zsock_new_push(endpoint.c_str());
    CHECK(push_!=nullptr);
    poller_=zpoller_new(sub_, NULL);
    CHECK(poller_!= nullptr);
  }else{
    sub_=push_=nullptr;
  }
}

void ParamManager::InitParams(){
  for(auto& entry: ownerid2Params_){
    entry.second.at(0)->Init();
  }
}
void ParamManager::Run(int step){
  running_=true;
  while(running_){
    Update(step++,0);
  }
}

void ParamManager::SyncWithPS(int step){
  if(sub_!=nullptr){
    zsock_t *which=static_cast<zsock_t*> (zpoller_wait(poller_,timeout_));
    int npull=0;
    int type, offset;
    while(which==sub_){
      zframe_t *frame=zframe_new_empty();
      CHECK_EQ(0, zsock_recv(sub_, "iif", &type, &offset, &frame));
      CHECK_EQ(type, kDataFrame);
      float* dptr=param_.mutable_cpu_data()+offset;
      float* inc=reinterpret_cast<float*>(zframe_data(frame));
      for(size_t j=0;j<zframe_size(frame)/sizeof(float);j++)
        dptr[j]+=inc[j];
      zframe_destroy(&frame);
      if(++npull>updateLimit_){
        LOG(ERROR)<<"Too many updates from parameter server, limit:"
          <<updateLimit_;
        break;
      }
      which=static_cast<zsock_t*>(zpoller_wait(poller_,timeout_));
    }
  }
  // TODO syn with parameter server
  if(step%syncfreq_==0){

  }
}

void ParamManager::Update(int step, int threadid){
  if(hogwild_){
    int gThreadid=Cluster::Get()->group_threadid(threadid);
    for(auto& layer: net_->layers()){
      if(layer->locationid()==gThreadid){
        for(Param* p: layer->GetParams())
          updater_->Update(step, p);
      }
    }
  }else{
    if(threadid==0){//main thread does update job
      for(auto& entry: ownerid2Params_){
        if(entry.second.size()>1){
          int len=entry.second.at(0)->data().count();
          float* accumgrad=entry.second.at(0)->mutable_cpu_grad();
          for(size_t k=1;k<entry.second.size();k++){
            float* grad=entry.second.at(k)->mutable_cpu_grad();
            for(int i=0;i<len;i++)
              accumgrad[i]+=grad[i];
          }
        }
        updater_->Update(step, entry.second.at(0), 1.0f/entry.second.size());
      }
      std::unique_lock<std::mutex> lck(mtx_);
      step_=step;
      cv_.notify_all();
    }else{
      // other threads wait
      std::unique_lock<std::mutex> lck(mtx_);
      while(step_<step) cv_.wait(lck);
    }
  }
}
}
