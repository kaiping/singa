// Copyright © 2014 Jinyang Gao. All Rights Reserved.
// 2014-07-18 16:33

#include <vector>

#include "core/table_delegate.h"
#include "core/disk-table.h"
#include "core/table.h"
#include "proto/model.pb.h"
#include "net/solver.h"


#include "da/dary.h"

DECLARE_bool(restore);
namespace lapis {
UpdateHandler<AdaGradValue>::UpdateHandler(const SolverProto& solver){
}
bool UpdateHandler<AdaGradValue>::Update(AdaGradValue* data,
    const AdaGradValue& update){
  int gid=update.gid();
  CHECK_GE(gid,0);
  int threshold=data->threshold();
  int n_update=data->n_update(gid);
  int len=data->data().value_size();
  if(data->grad_size()==0){
    for(int i=0;i<GlobalContext::Get()->num_groups();i++){
      DAryProto* grad=data->add_grad();
      for(int j=0;j<len;j++)
        grad->add_value(0.0f);
    }
  }
  if(data->history().value_size()==0){
    DAryProto* hist=data->mutable_history();
    for(int i=0;i<len;i++)
      hist->add_value(data->kinit());
  }

  int offset=0;
  float* graddptr=data->mutable_grad(gid)->mutable_value()->mutable_data();
  //threshold is the num of workers in one group contributing to this grad
  if(n_update<threshold){
    for(auto v: update.grad(0).value())
      graddptr[offset++]+=v;
    CHECK_EQ(offset, len);
  }

  data->set_n_update(gid, n_update+1);
  if(data->n_update(gid)==threshold){
    //basic adagrad
    float * history=data->mutable_history()->mutable_value()->mutable_data();
    float * dptr=data->mutable_data()->mutable_value()->mutable_data();
    float lr=data->learning_rate();
    for(int i=0;i<len;i++){
      history[i]+=graddptr[i]*graddptr[i];
      dptr[i]-=graddptr[i]*lr/sqrt(history[i]);
      graddptr[i]=0.f;
    }
    data->set_n_update(gid, 0);
    data->set_version(data->version()+1);
  }
  return true;
}
bool UpdateHandler<AdaGradValue>::Get(const VKey& key,
    const AdaGradValue &val, AdaGradValue* ret) {
  //LOG(INFO)<<"key version "<<key.version()<<" val version "<<val.version();
  int gid=key.gid();
  CHECK_GE(gid,0);
  if(key.version()<=val.version()&&val.n_update(gid)==0){
    DAryProto* retdat=ret->mutable_data();
    retdat->clear_value();
    for(auto x: val.data().value())
      retdat->add_value(x);
    return true;
  }else
    return false;
}

bool UpdateHandler<AdaGradValue>::is_checkpointable(const VKey& key,
    const AdaGradValue& val) {
  /*
  if(val.version()>FLAGS_checkpoint_after&&
      val.version()%FLAGS_checkpoint_frequency==0)
    return true;
  else
    */
    return false;
}

/*********************************************************************
 * SGDValue
 **********************************************************************/
bool UpdateHandler<SGDValue>::is_checkpointable(const VKey& key, const SGDValue& val) {
  /*
  if(val.version()>FLAGS_checkpoint_after&&
      val.version()%FLAGS_checkpoint_frequency==0)
    return true;
  else
  */
    return false;
}

bool UpdateHandler<SGDValue>::Get(const VKey& key, const SGDValue &val, SGDValue* ret) {
  //LOG(INFO)<<"key version "<<key.version()<<" val version "<<val.version();
  if(key.version()<=val.version()){
    DAryProto* retdat=ret->mutable_data();
    retdat->clear_value();
    for(auto x: val.data().value())
      retdat->add_value(x);
    return true;
  }
  return false;
}


UpdateHandler<SGDValue>::UpdateHandler(const SolverProto& solver){
  step_=0;
  base_learning_rate_=solver.sgd().base_learning_rate();
  gamma_=solver.sgd().gamma();
  learning_rate_change_=solver.sgd().learning_rate_change();
  learning_rate_change_steps_=solver.sgd().learning_rate_change_steps();
  momentum_=solver.sgd().momentum();
  weight_decay_=solver.sgd().weight_decay();
}

bool UpdateHandler<SGDValue>::Update(SGDValue* data, const SGDValue& update){
  CHECK_EQ(data->version(), update.version())<<data->id()<<" "<<data->threshold()<<" "<<data->n_update();
  int len=data->data().value_size();
  if(data->history().value_size()==0){
    DAryProto* hist=data->mutable_history();
    for(int i=0;i<len;i++)
      hist->add_value(0.0f);
  }

  CHECK_EQ(len, update.grad(0).value_size());
  CHECK_EQ(len, data->history().value_size());

  float* history=data->mutable_history()->mutable_value()->mutable_data();
  const float* grad=update.grad(0).value().data();
  float* dptr=data->mutable_data()->mutable_value()->mutable_data();
  float lr=learning_rate_*data->learning_rate_multiplier();
  float w=weight_decay_*data->weight_decay_multiplier();
  // hist=hist-lr*grad
  DAry::arymath().madd(history, lr, grad, history, len);
  // hist=hist-lr*weight*param
  if(w>0)
    DAry::arymath().madd(history, lr*w, dptr, history, len);

  data->set_n_update(data->n_update()+1);
  if(data->n_update()==data->threshold()){
    // param+=history/n, /data->n_update()
    //DAry::arymath().sub(dptr, dptr, history, len);
    float factor=-1.0/GlobalContext::Get()->num_groups();
    DAry::arymath().madd(dptr, factor, history, dptr, len);
    // hist=hist*mom
    DAry::arymath().mul(history, momentum_, history, len);
    data->set_n_update(0);
    data->set_version(update.version()+1);
    UpdateHyperParams(data->version());
  }
  return true;
}

void UpdateHandler<SGDValue>::UpdateHyperParams(const int step) {
  learning_rate_ = Solver::UpdateHyperParam(step, learning_rate_change_,
      learning_rate_change_steps_,
      base_learning_rate_,
      gamma_);
}

/********************************************************************
 * Table Delegate
 * *************************************************************/
void TableDelegate::HandleShardAssignment() {
  LOG(INFO) << "Handle Shard Assignment";
  ShardAssignmentRequest shard_req;
  auto mpi=NetworkThread::Get();
  mpi->Read(GlobalContext::kCoordinator, MTYPE_SHARD_ASSIGNMENT, &shard_req);
  auto context=GlobalContext::Get();
  //  request read from coordinator
  auto _tables=tables();
  for (int i = 0; i < shard_req.assign_size(); i++) {
    const ShardAssignment &a = shard_req.assign(i);
    GlobalTable *t = _tables.at(a.table());
    t->get_partition_info(a.shard())->owner = a.new_worker();
    //if local shard, create check-point files
    if (context->checkpoint_enabled() && t->is_local_shard(a.shard())){
      string checkpoint_file = StringPrintf("%s/checkpoint_%d",context->data_folder().data(), a.shard());
      FILE *tmp_file = fopen(checkpoint_file.c_str(), "r");
      if (tmp_file){//exists -> open to reading and writing
        fclose(tmp_file);
        auto cp = t->checkpoint_files();

        if (FLAGS_restore){//open in read mode to restore, then close
          (*cp)[a.shard()] = new LogFile(checkpoint_file,"r",a.shard());
          t->Restore(a.shard());
          delete (*cp)[a.shard()];
          EmptyMessage dummy;
          mpi->Send(GlobalContext::kCoordinator, MTYPE_SERVER_RESTORED, dummy);
          LOG(ERROR) << "Server restored";
        }

        VLOG(3) << "Open checkpoint file for writing";
        (*cp)[a.shard()] = new LogFile(checkpoint_file,"a",a.shard());
      }
      else{// not exist -> open to writing first time
        auto cp = t->checkpoint_files();
        (*cp)[a.shard()] = new LogFile(checkpoint_file,"w",a.shard());
        VLOG(3) << "Added to new checkpoint files for shard "<< a.shard();
      }

    }
  }
  EmptyMessage empty;
  mpi->Send(GlobalContext::kCoordinator, MTYPE_SHARD_ASSIGNMENT_DONE, empty);
  LOG(ERROR)<<"Finish handle shard assignment";
}

TableDelegate* CreateTableDelegate(const SolverProto& proto){
/**
 * need to know the tuple type to create parameter table
 */
  TableDelegate *delegate;
  if(proto.method()==lapis::SolverProto::kSGD){
     delegate=new lapis::TypedTableDelegate<VKey, lapis::SGDValue>(proto);
  }
  else{
    delegate= new lapis::TypedTableDelegate<VKey, lapis::AdaGradValue>(proto);
  }
  NetworkThread::Get()->RegisterCallback(MTYPE_SHARD_ASSIGNMENT,
                         boost::bind(&TableDelegate::HandleShardAssignment, delegate));

  return delegate;
}


template<>
TypedTableDelegate<VKey,AdaGradValue>::TypedTableDelegate (const SolverProto& proto){
  auto* update_handler=new UpdateHandler<AdaGradValue>(proto);
  param_table_= CreateParamTable(0, GlobalContext::Get()->num_table_servers(),
      update_handler,new VKeySharder, new Marshal<VKey>, new Marshal<AdaGradValue>);

  example_=proto.adagrad();
  kMaxSplits_=proto.max_splits();
}


template<>
TypedTableDelegate<VKey,SGDValue>::TypedTableDelegate (const SolverProto& proto){
  auto* update_handler=new UpdateHandler<SGDValue>(proto);
  param_table_= CreateParamTable(0, GlobalContext::Get()->num_table_servers(),
      update_handler,new VKeySharder, new Marshal<VKey>, new Marshal<SGDValue>);

  example_=proto.sgd();
  kMaxSplits_=proto.max_splits();
}

void TableDelegate::Update(const std::vector<Param*> &params, int step) {
  for(auto* param: params)
    Update(param,step);
  return;
}

void TableDelegate::Put(const std::vector<Param*> &params) {
  for(auto* param: params)
    Put(param);
}

void TableDelegate::Get(const std::vector<Param*> &params, int step){
  for(auto* param : params)
    Get(param, step);
  return;
}
void TableDelegate::AsyncGet(const std::vector<Param*> &params, int step){
  for(auto* param : params)
    AsyncGet(param, step);
  return;
}

template<>
void TypedTableDelegate<VKey, SGDValue>::Put(Param * param){
  int offset = 0;
  int nworkers=GlobalContext::Get()->num_workers();
  int ngroups=GlobalContext::Get()->num_groups();
  const float * data_addr = param->data().dptr();
  for(auto& entry: param_splits_[param->id()]) {
    SGDValue v(example_);
    // sgd related hyper-parameters
    v.set_learning_rate_multiplier(param->learning_rate_multiplier());
    v.set_weight_decay_multiplier(param->weight_decay_multiplier());
    v.set_version(0);
    v.set_n_update(0);
    v.set_id(param->id());
    if(!param->partition())
      v.set_threshold(nworkers);
    else
      v.set_threshold(ngroups);
    LOG(INFO)<<"param "<<v.id()<<" threshold "<<v.threshold()<<" "<<nworkers<<" "<<ngroups;
    DAryProto* dary=v.mutable_data();
    dary->clear_value();
    for(int k = 0; k < entry.second; k++){
      dary->add_value(data_addr[offset]);
      offset++;
    }
    VKey key;
    key.set_version(0);
    key.set_key(entry.first);
    param_table_->put(key, v);
  }
}

template<>
void TypedTableDelegate<VKey, AdaGradValue>::Put(Param * param){
  int offset = 0;
  const float * data_addr = param->data().dptr();
  int groupsize=GlobalContext::Get()->group_size();
  int ngroups=GlobalContext::Get()->num_groups();
  for(auto& entry: param_splits_[param->id()]) {
    AdaGradValue v(example_);
    // sgd related hyper-parameters
    v.set_version(0);
    if(!param->partition())
      v.set_threshold(groupsize);
    else
      v.set_threshold(1);
    for(int i=0;i<ngroups;i++)
      v.add_n_update(0);
    DAryProto* dary=v.mutable_data();
    dary->clear_value();
    for(int k = 0; k < entry.second; k++){
      dary->add_value(data_addr[offset]);
      offset++;
    }
    VKey key;
    key.set_version(0);
    key.set_key(entry.first);
    param_table_->put(key, v);
  }
}


}  // namespace lapis

