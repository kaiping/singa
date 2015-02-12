#include "worker/param_manager.h"

namespace singa{

void ParamManager::Setup(shared_ptr<NeuralNet> net,
    shared_ptr<ParamUpdater> updater){
  int count=0;
  updater_=updater;
  shared_ptr<Cluster> cluster=Cluster::Get();
  for(shared_ptr<Layer> layer: net->layers()){
    if(layer->locationid()==-1||cluster->InMyProcs(layer->locationid())){
      for(Param* p: layer->params()){
        int ownerid=p->owner()->id();
        if(paramOffset_.find(ownerid)==paramOffset_.end()){
          //ownerID2procsID_[ownerid]=cluster->myprocsID();
          aggregatedUpdates_[ownerid]=0;
          paramOffset_[p->id()]=count;
          count+=p->count();
        }else{
          paramOffset_[p->id]=paramOffset_[ownerid];
        }
        ownerID2Params[ownerid].push_back(p);
        paramID2param_[p->id()]=p;
      }
    }
  }

  param_.Reshape(vector<int> {count});
  const float* dptr=param_->cpu_data();
  for(auto& entry: paramID2param_){
    entry.second->mutable_data()->data()->set_cpu_data(
        dptr+paramOffset_[entry.second]);
  }
  // for inproc pub and pull
  pub_=zsock_new_pub("@inproc://pmpub");
  pull_=zsocket_new_pull("@inproc://pmpull");
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
  for(auto& entry: paramID2param_){
    Param* p=entry->second;
    if(p->owner()==p){// haven't initialized
      p->Init();
      zscock_send(pub_, "ii", kParamReady, p->ID());
    }
  }
}

void ParamManager::Run(int step){
  running_=true;
  while(running_){
    Update(step++);
  }
}

void ParamManager::Update(int step){
  for(size_t k=0;k<paramID2param_.size();k++){
    int type=0, paramid;
    Param* param=nullptr;
    CHECK_EQ(0, zsock_recv(pull_, "ip", &type, &param));
    CHECK_EQ(type, kGradReady);
    //CHECK(paramID2param_.find(paramid)!=paramID2param_.end());
    int owner=param->owner()->id();
    aggregatedUpdates_[owner]++;
    const vector<Param*>& params=ownerID2Params_[owner];
    CHECK_LE(aggregatedUpdates_[owner], params.size());
    if(aggregatedUpdates_[owner]==params.size()){
      float* accumgrad=params[0]->mutable_cpu_grad();
      if(params.size()>1){
        for(int i=1;i<params.size();i++){
          float* grad=params[1]->mutable_cpu_grad();
          for(int j=0;j<params[0]->count();j++)
            accumgrad[j]+=grad[j];
        }
        for(int j=0;j<params[0]->count();j++)
          accumgrad[j]/=params.size();
      }
      updater_.Update(step, params[0]);
      aggregatedUpdates_[owner]=0;
    }
  }

  if(sub_!=nullptr){
    zsock_t *which=static_cast<zsock_t*> zpoller_wait(poller_,timeout_);
    int npull=0;
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
          <updateLimit_;
        break;
      }
      which=static_cast<zsock_t*> zpoller_wait(poller_,timeout_);
    }
  }

  type=kDataReady;
  zscock_send(pub_, "ii", &type, );

  // TODO syn with parameter server
  if(step%syncfreq_==0){

  }
}

}
