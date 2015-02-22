#include "utils/cluster.h"
#include "worker/param_manager.h"
#include "utils/singleton.h"
#include "utils/factory.h"


namespace singa{

ParamManager::ParamManager(shared_ptr<NeuralNet> net,
    const UpdaterProto& updater):net_(net){
  shared_ptr<Cluster> cluster=Cluster::Get();
  hogwild_=cluster->nthreads_per_procs()==1||updater.hogwild()?true:false;
  sync_frequency_=updater.sync_frequency();
  warmup_steps_=updater.warmup_steps();
  sample_ratio_=updater.sample_ratio();
  switch(updater.type()){
    case UpdaterProto_Type_kAdaGrad:
    updater_=make_shared<AdaGradUpdater>();
    break;
    case UpdaterProto_Type_kAdaDelta:
    updater_=make_shared<AdaDeltaUpdater>();
    break;
    case UpdaterProto_Type_kNesterov:
    updater_=make_shared<NesterovUpdater>();
    break;
    case UpdaterProto_Type_kSGD:
    updater_=make_shared<SGDUpdater>();
    break;
    case UpdaterProto_Type_kRMSProp:
    updater_=make_shared<RMSPropUpdater>();
    break;
    default:
    LOG(FATAL)<<"Unknow updater "<<updater.type();
  }
  updater_->Init(updater);

  int count=0;
  for(shared_ptr<Layer> layer: net->layers()){
    if(cluster->group_procsid(layer->locationid())==cluster->group_procsid()){
      for(shared_ptr<Param> p: layer->GetParams()){
        int ownerid=p->owner()->id();
        if(paramid2Offset_.find(ownerid)==paramid2Offset_.end()){
          paramid2Offset_[p->id()]=count;
          paramid2Offset_[ownerid]=count;
          count+=p->data().count();
        }else{
          paramid2Offset_[p->id()]=paramid2Offset_[ownerid];
        }
        ownerid2Params_[ownerid].push_back(p);
        if(ownerid2Params_[ownerid].size()>1)
          aggregatedUpdates_[ownerid]=0;
        paramid2version_[p->id()]=0;
        paramid2Param_[p->id()]=p;
      }
    }
  }

  ParamProto pp;
  Factory<Param>* factory=Singleton<Factory<Param>>::Instance();
  param_=shared_ptr<Param>(factory->Create("Param"));
  param_->Setup(pp, vector<int> {count});
  float* dptr=param_->mutable_cpu_data();
  for(auto& entry: ownerid2Params_){
    entry.second.at(0)->data().data()->set_cpu_data(
        dptr+paramid2Offset_[entry.first]);
  }

  if(cluster->nservers()>0){ // sync with parameter server
    router_=make_shared<Router>(cluster->router_port());
    for(int i=0;i<cluster->nservers();i++)
      CHECK(router_->Connect(cluster->server_addr(i)));
  }
}

ParamManager::~ParamManager(){
  for(int i=0;i<Cluster::Get()->nservers();i++){
    zmsg_t* msg=zmsg_new();
    zmsg_addstrf(msg, "%d", kStop);
    router_->Send(msg, i);
  }
  zclock_sleep(2000);
}


void ParamManager::SyncConfig(float compute_time){
  float modelsize=param_->size()*1.0f*sizeof(float)/1024/1024; //MB
  auto cluster=Cluster::Get();
  float cputhroughput=1.0f*modelsize*cluster->nworkers()/compute_time; //MB/s
  sample_ratio_=cluster->bandwidth()*cluster->nservers()/cputhroughput;
  if(sample_ratio_>1.0f)
    sample_ratio_=1.0f;
  LOG(ERROR)<<"Sample Ratio "<<sample_ratio_;
}

void ParamManager::InitParams(){
  for(auto& entry: ownerid2Params_){
    entry.second.at(0)->Init();
  }
}
void ParamManager:: SendParamsToServers(){
  for(auto &entry: ownerid2Params_){
    for(shared_ptr<Param> p:entry.second){
      int id=p->id();
      zmsg_t* msg=zmsg_new();
      zmsg_addstrf(msg, "%d", kPut);
      zmsg_addstrf(msg,"%d",id);
      zmsg_addstrf(msg,"%s",p->name().c_str());
      zmsg_addmem(msg, p->mutable_cpu_data(), sizeof(float)*p->data().count());
      router_->Send(msg, id%Cluster::Get()->nservers());
      if(!hogwild_)
        break;
    }
  }
}

void ParamManager::GetParamsFromServers(int step){// will be blocked until recv all parameters.
  for(auto &entry: ownerid2Params_){
    int id=entry.first;
    for(shared_ptr<Param> p:entry.second){
      zmsg_t* msg=zmsg_new();
      zmsg_addstrf(msg, "%d", kGet);
      zmsg_addstrf(msg,"%d",id);
      zmsg_addstrf(msg,"%s",p->name().c_str());
      router_->Send(msg, id%Cluster::Get()->nservers());
      if(!hogwild_)
        break;
    }
  }
  size_t nrecv=0, ntotal=hogwild_?paramid2version_.size():ownerid2Params_.size();
  int id, type;
  while(nrecv<ntotal){
    zmsg_t* msg=router_->Recv();
    char* typestr=zmsg_popstr(msg); sscanf(typestr, "%d", &type); delete typestr;
    CHECK_EQ(type, kGet);
    char* idstr=zmsg_popstr(msg);  sscanf(idstr, "%d", &id); delete idstr;
    CHECK(paramid2Param_.find(id)!=paramid2Param_.end());
    char* name=zmsg_popstr(msg); //name
    CHECK_STREQ(name, paramid2Param_[id]->name().c_str());
    delete name;
    zframe_t* dat=zmsg_pop(msg);
    shared_ptr<Param> p=paramid2Param_[id];
    CHECK_EQ(zframe_size(dat), p->data().count()*sizeof(float));
    memcpy(p->mutable_cpu_data(), zframe_data(dat), zframe_size(dat));
    zframe_destroy(&dat);
    zmsg_destroy(&msg);
    nrecv++;
    paramid2version_[p->id()]=step;
    if(!hogwild_){
      for(shared_ptr<Param>p: ownerid2Params_.at(p->owner()->id())){
        paramid2version_[p->id()]=step;
      }
    }
  }
}
bool ParamManager::SyncNow(int step){
  return Cluster::Get()->nservers()
    &&(step+1)%sync_frequency_==0
    &&step>warmup_steps_;
}
void ParamManager::UpdateParam(shared_ptr<Param> param, int step, int local_threadid){
  bool sync=SyncNow(step+1);
  if(hogwild_||ownerid2Params_[param->owner()->id()].size()==1){
    updater_->Update( step, param);
    paramid2version_[param->id()]=step+(sync==false);
  }else{
    bool update=false;
    const auto& shares =ownerid2Params_[param->owner()->id()];
    {
      std::unique_lock<std::mutex> lck(mtx_);
      aggregatedUpdates_[param->owner()->id()]++;
      update=aggregatedUpdates_[param->owner()->id()] == shares.size();
    }
    if(update){
      paramid2version_[shares.at(0)->id()]=step+1;
      float* accumgrad=shares.at(0)->mutable_cpu_grad();
      int len=shares.at(0)->data().count();
      for(size_t k=1;k<shares.size();k++){
        paramid2version_[shares.at(k)->id()]=step+(sync==false);
        float* grad=shares.at(k)->mutable_cpu_grad();
        for(int i=0;i<len;i++)
          accumgrad[i]+=grad[i];
      }
      updater_->Update(step,shares.at(0), 1.0f/shares.size());
      aggregatedUpdates_[param->owner()->id()]=0;
      param=shares.at(0);
    }else
      sync=false;
  }
  if(sync){
    zmsg_t *msg=param->GenSyncMsgFromWorker(sample_ratio_);
    zmsg_pushstrf(msg, "%d", param->id());
    zmsg_pushstrf(msg, "%d", kSync);
    router_->Send(msg, param->id()%Cluster::Get()->nservers());
  }
}

void ParamManager::WaitUpdate(shared_ptr<Param> param, int step, int local_threadid){
  if(SyncNow(step)){
    while(paramid2version_[param->id()]<step){
      zmsg_t* msg=router_->Recv();
      char* typestr=zmsg_popstr(msg);
      int type;
      sscanf(typestr, "%d", &type);
      delete typestr;
      CHECK_EQ(type, kSync);

      char* idstr=zmsg_popstr(msg);
      int id;
      sscanf(idstr, "%d", &id);
      delete idstr;

      CHECK(paramid2Param_.find(id)!=paramid2Param_.end());
      paramid2Param_[id]->ParseSyncMsgFromPS(msg);
      zmsg_destroy(&msg);
      paramid2version_[id]=step;
      int ownerid=paramid2Param_[id]->owner()->id();
      if(!hogwild_){
        for(shared_ptr<Param>p: ownerid2Params_[ownerid]){
          paramid2version_[p->id()]=step;
        }
      }
    }
  }
    // wait to recv param ready singal
  while(paramid2version_[param->id()]<step)
    Sleep(5);
}
/*
      std::unique_lock<std::mutex> lck(mtx_);
      step_=step;
      cv_.notify_all();
    }else{
      // other threads wait
      std::unique_lock<std::mutex> lck(mtx_);
      while(step_<step) cv_.wait(lck);
    }
*/
}
