#include "worker/param_manager.h"

namespace singa{

ParamManager::ParamManager(shared_ptr<NeuralNet> net,
    const UpdaterProto& updater){
  step_=-1;
  int count=0;
  if(updater.handler()==ParamUpdater_Type_kAdaGrad){
    updater_=make_shared<AdaGradUpdater>();
    updater_->Init(updater);
  } else
    LOG(FATAL)<<"Only support AdaGradUpdater now";

  shared_ptr<Cluster> cluster=Cluster::Get();
  for(shared_ptr<Layer> layer: net->layers()){
    int threadID=layer->locationID();
    int procsID=cluster_->procsID_of_thread(threadID);
    if(procsID==cluster_->procsID()){
      for(Param* p: layer->params()){
        int ownerid=p->owner()->ID();
        if(paramID2Offset_.find(ownerid)==paramID2Offset_.end()){
          paramID2Offset_[p->id()]=count;
          paramID2Offset_[ownerid]=count;
          count+=p->count();
        }else{
          paramID2Offset_[p->id]=paramID2Offset_[ownerid];
        }
        ownerID2Params_[ownerid].push_back(p);
        //paramID2param_[p->id()]=p;
      }
    }
  }

  param_.Reshape(vector<int> {count});
  const float* dptr=param_->cpu_data();
  for(auto& entry: ownerID2Params_){
    entry.second.at(0)->data().data()->set_cpu_data(
        dptr+paramID2Offset_[entry.first]);
  }
  if(cluster->num_servers()>0){ // sync with parameter server
    string ps= cluster->server_addr(
        cluster_->group_procsID(cluster_->procsID()));
    sub_=zsock_new_sub(">tcp://"+ps+":"+cluster_->pub_port());
    CHECK_NE(sub_,nullptr);
    push_=zsock_new_push(">tcp://"+ps+":"+cluster_->pull_port());
    CHECK_NE(push_,nullptr);
    pooler_=zpoller_new(sub_, NULL);
    CHECK_NE(poller_, nullptr);
  }else{
    sub_=push_=nullptr;
  }
}

void ParamManager::InitParams(){
  for(auto& entry: ownerID2Params_){
    entry.second.at(0)->Init();
  }
  std::unique_lock<std::mutex> lck(mtx_);
  step_=step;
  cv.notify_all();
}

void ParamManager::Run(int step){
  running_=true;
  while(running_){
    Update(step++);
  }
}

void ParamManager::SyncWithPS(int step){
  if(sub_!=nullptr){
    zsock_t *which=static_cast<zsock_t*> zpoller_wait(poller_,timeout_);
    int npull=0;
    int type, offset;
    while(which==pull_){
      zframe_t frame=zframe_new_empty();
      CHECK_EQ(0, zsock_recv(sub_, "iif", &type, &offset, &frame));
      CHECK_EQ(type, kDataFrame);
      float* dptr=param_->mutable_cpu_data()+offset;
      float* inc=zframe_data(frame);
      for(int j=0;j<zframe_size(frame)/sizeof(float);j++)
        dptr[j]+=inc[j];
      zframe_destroy(&frame);
      if(++npull>updateLimit_){
        LOG(ERROR)<<"Too many updates from parameter server, limit:"
          <<updateLimit_;
        break;
      }
      which=static_cast<zsock_t*> zpoller_wait(poller_,timeout_);
    }
  }
  // TODO syn with parameter server
  if(step%syncfreq_==0){

  }
}

void ParamManager::Update(int step, shared_ptr<NeuralNet> net,  int threadID){
  if(updater_.hogwild()){
    int gThreadID=cluster_->group_threadID(threadID);
    for(auto& layer: net->layers()){
      if(layer->locationID()==-1||layer->locationID()==gThreadID){
        for(Param* p: layer->GetParams())
          updater_.Update(step, p);
      }
    }
  }else{
    if(threadID==0){//main thread does update job
      for(auto& entry: ownerID2Params_){
        if(entry.second.size()>1){
          int len=entry.second.at(0)->data().count();
          float* accumgrad=entry.second.at(0)->mutable_cpu_grad();
          for(size_t k=1;k<entry.second.size();k++){
            float* grad=entry.second.at(k)->mutable_cpu_grad();
            for(int i=0;i<len;i++)
              accumgrad[i]+=grad[i];
          }
        }
        updater_.Update(step, entry.second.at(0), 1.0f/entry.second.size());
      }
      std::unique_lock<std::mutex> lck(mtx_);
      step_=step;
      cv.notify_all();
    }else{
      // other threads wait
      std::unique_lock<std::mutex> lck(mtx_);
      while(step_<step) cv_.wait(lck);
    }
  }
}
}
