#include "utils/cluster.h"
#include "worker/param_manager.h"

namespace singa{

ParamManager::ParamManager(shared_ptr<NeuralNet> net,
    const UpdaterProto& updater):net_(net){
  shared_ptr<Cluster> cluster=Cluster::Get();
  hogwild_=cluster->nthreads_per_procs()==1||updater.hogwild()?true:false;
  sync_frequency_=updater.sync_frequency();
  if(updater.type()==UpdaterProto_Type_kAdaGrad){
    updater_=make_shared<AdaGradUpdater>();
    updater_->Init(updater);
  } else
    LOG(FATAL)<<"Only support AdaGradUpdater now";

  int count=0;
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
        if(ownerid2Params_[ownerid].size()>1)
          aggregatedUpdates_[ownerid]=0;
        param2version_[p]=0;
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
  }
}

void ParamManager::InitParams(){
  for(auto& entry: ownerid2Params_){
    entry.second.at(0)->Init();
  }
}

void ParamManager::UpdateParam(Param* param, int step, int local_threadid){
  bool sync=true;
  if(hogwild_||ownerid2Params_[param->owner()->id()].size()==1){
    updater_->Update( step, param);
    param2version_[param]=step+1;
  }else{
    const vector<Param*>& shares=ownerid2Params_[param->owner()->id()];
    {
      std::unique_lock<std::mutex> lck(mtx_);
      aggregatedUpdates_[param->owner()->id()]++;
      sync=aggregatedUpdates_[param->owner()->id()] == shares.size();
    }
    if(sync){
      param2version_[shares.at(0)]=step+1;
      float* accumgrad=shares.at(0)->mutable_cpu_grad();
      int len=shares.at(0)->data().count();
      for(size_t k=1;k<shares.size();k++){
        param2version_[shares.at(k)]=step+1;
        float* grad=shares.at(k)->mutable_cpu_grad();
        for(int i=0;i<len;i++)
          accumgrad[i]+=grad[i];
      }
      updater_->Update(step,shares.at(0), 1.0f/shares.size());
      aggregatedUpdates_[param->owner()->id()]=0;
    }
  }
  if(sync&&Cluster::Get()->nservers()&&(step+1)%sync_frequency_==0){
    zmsg_t *msg=param->GenSyncMsgFromWorker();
    // send msg;
  }
}

void ParamManager::WaitUpdate(Param* param, int step, int local_threadid){
  if(Cluster::Get()->nservers()&&step%sync_frequency_==0){

  }
    // wait to recv param ready singal
  while(param2version_[param]<step)
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
