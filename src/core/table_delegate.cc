// Copyright © 2014 Wei Wang & Jinyang Gao. All Rights Reserved.
// 2014-07-18 16:33

#include <vector>

#include "core/table_delegate.h"
#include "core/table.h"
#include "proto/model.pb.h"
#include "net/solver.h"


#include "da/dary.h"

DECLARE_bool(restore);
namespace lapis {
TableDelegate::~TableDelegate(){
}

void TableDelegate::SplitParams(const vector<Param *>& params) {
  int total_splits=0;
  collect_counter_=new std::atomic<int>[params.size()];
  for(auto param: params){
    SplitParam(param, GlobalContext::Get()->id());
  }
  LOG(INFO)<<"Total splits for this worker "<<total_splits;
}

void TableDelegate::SplitParam(const Param & param, int worker_id){
  int group_size=GlobalContext::Get()->group_size();
  const DAry& dary=param->data();
  int id=param->id()*group_size;
  if(param->partition())
    id+=worker_id;
  int local_size=dary.local_size();
  int splitsize=param->split_threshold();
  if(splitsize>=16777216){
    LOG(WARNING)<<"split of size "<<splitsize
      <<"  exceeds the size of max google protobuf message, i.e., 64MB"
      <<" param length is "<<local_size
      <<", reset the split threshold to 4000,000 Bytes";
    splitsize = 1000000;
  }
  int nsplits=local_size/splitsize+(local_size%splitsize!=0);
  CHECK_LE(nsplits,kMaxSplits_)<<"total splits for one param partition "
    <<" exceeds kMaxSplits, raise kMaxSplits in solver config";
  vector<std::pair<int, int>> splits;
  for(auto j = 0, pos=0; j < nsplits; j++) {
    int len=(pos+splitsize)<local_size?splitsize:local_size-pos;
    int splitid=id*kMaxSplits_+j;
    splits.push_back(std::make_pair(splitid,len));
    split_map_[splitid]=std::make_pair(param, pos);
    asyncget_split_[splitid]=false;
    pos+=len;
  }
  CHECK_EQ(param_splits_.size(), param->id());
  splits_.push_back(splits);
  total_splits+=splits.size();
  // for debug
  for(auto& split: splits){
    char tmpbuf[1024];
    sprintf(tmpbuf, "%4d %5d %5d", param->id(), split.first, split.second);
    LOG(INFO)<<string(tmpbuf);
  }
}
void TableDelegate::Update(const std::vector<Param*> &params, int step) {
  for(auto* param: params)
    Update(param,step);
  return;
}

void TableDelegate::Update(Param *param, int step){
  TKey key;
  key.set_version(step);
  int offset = 0;
  const float * dptr = param->grad().dptr();
  for(auto& entry: splits_[param->id()]) {
    TVal val;
    DAryProto* grad=val.mutable_grad();

    // sgd related hyper-parameters
    for(int k = 0; k < entry.second; k++){
      grad->add_value(dptr[offset++]);
    }

    key.set_id(entry.first);
    std::string valstr=val.SerializeToString();
    TableData tuple;
    tuple.set_shard(Shard(entry.first,GlobalContext::Get()->num_servers()));
    tuple.set_source(GlobalContext::Get()->rank());
    tuple.set_table(0);
    Arg *arg=tuple.add_kv_data();
    arg->set_key(key.SerializeToString());
    arg->set_allocated_value(&valstr);
    tuple.set_done(true);
    NetworkThread::Get()->Send(tuple.shard(), MTYPE_UPDATE_REQUEST, tuple);
    //Network::Get()->Send(tuple.shard(), MTYPE_UPDATE_REQUEST, tuple);
  }
}


void TableDelegate::Get(const std::vector<Param*> &params, int step){
  for(auto* param : params)
    Get(param, step);
  return;
}

/*
void TableDelegate::Get(Param * param, int step){
  float* dptr=param->mutable_data()->dptr();
  TKey key;
  key.set_version(step);
  int offset=0;
  for(auto entry: splits_[param->id()]) {
    key.set_id(entry.first);
    TVal v=table_->get(key);
    for(auto x: val.data().value()){
      dptr[offset++]=x;
    }
    CHECK_EQ(val.data().value_size(), entry.second);
  }
  CHECK_EQ(offset, param->data().local_size());
}
*/
void TableDelegate::AsyncGet(const std::vector<Param*> &params, int step){
  for(auto* param : params)
    AsyncGet(param, step);
  return;
}
void TableDelegate::Put(const std::vector<Param*> &params) {
  for(auto* param: params)
    Put(param);
}
void TableDelegate::Put(Param * param){
  int offset = 0;
  int nworkers=GlobalContext::Get()->num_workers();
  const float * data_addr = param->data().dptr();
  for(auto& entry: param_splits_[param->id()]) {
    TVal val;
    // sgd related hyper-parameters
    val.set_learning_rate_multiplier(param->learning_rate_multiplier());
    val.set_weight_decay_multiplier(param->weight_decay_multiplier());
    val.set_param_id(param->id());
    val.set_splitid(entry.first);
    val.set_splitoffset(split_map_[entry.first]);
    DAryProto* dary=val.mutable_data();
    for(int k = 0; k < entry.second; k++){
      dary->add_value(data_addr[offset]);
      offset++;
    }
    TKey key;
    key.set_version(0);
    key.set_id(entry.first);
    std::string valstr=val.SerializeToString();
    TableData tuple;
    tuple.set_shard(Shard(entry.first,GlobalContext::Get()->num_servers()));
    tuple.set_source(GlobalContext::Get()->rank());
    tuple.set_table(0);
    Arg *arg=tuple.add_kv_data();
    arg->set_key(key.SerializeToString());
    arg->set_allocated_value(&valstr);
    tuple.set_done(true);
    NetworkThread::Get()->Send(tuple.shard(), MTYPE_UPDATE_REQUEST, tuple);
    //Network::Get()->Send(tuple.shard(), MTYPE_UPDATE_REQUEST, tuple);
  }
}

void TableDelegate::AsyncGet(Param * param, int step){
  TKey key;
  key.set_version(step);
  for(auto entry: splits_[param->id()]) {
    key.set_id(entry.first);
    HashGet req;
    req.set_key(key.SerializeToString());
    req.set_table(0);
    req.set_shard(Shard(entry.first, GlobalContext::Get()->num_servers()));
    req.set_source(GlobalContext::Get()->rank());
    NetworkThread::Get()->Send(req.shard(), MTYPE_GET_REQUEST, req);
    asyncget_split_.at(entry.first)=false;
    //LOG(INFO)<<"get "<<entry.first;
  }
}
void TableDelegate::AsyncCollect(Param * param, int step){
  auto& splits=param_splits_[param->id()];
  unsigned int nget=0;
  int start_split_id=splits.front().first;
  int end_split_id=splits.back().first;
  // check num of splits collected before
  for(auto& split: splits){
    if(asyncget_split_.at(split.first))
      nget++;
  }
  TKey key;
  TVal val;
  while(nget<splits.size()){
   // may collect splits of other params used later
    TableData tuple;
    if(NetworkThread::Get()->TryRead(MPI::ANY_SOURCE, MTYPE_GET_RESPONSE, &tuple)){
      key.ParseFromString(tuple.kv_data(0).key());
      val.ParseFromString(tuple.kv_data(0).value());
      int splitid=key.id();
      //LOG(INFO)<<"collected "<<splitid;
      auto& split=split_map_.at(splitid);
      Param* p=split.first;
      int offset=split.second;
      float * dptr = p->mutable_data()->dptr();
      for(auto v: val.data().value())
        dptr[offset++]=v;
      //val.mutable_data()->clear_value();
      // check this split is complete, i.e. offset is the start of next split
      if(split_map_.find(key.key()+1)!=split_map_.end())
        CHECK_EQ(offset, split_map_.at(key.key()+1).second);
      asyncget_split_[splitid]=true;
      if(splitid>=start_split_id&&splitid<=end_split_id)
        nget++;
    }else{
      sleep(0.0001);
    }
  }
  // check all splits have been collected, reset async get markers,
  for(auto& split:splits){
    CHECK(asyncget_split_[split.first]);
    asyncget_split_[split.first]=false;
  }
}


/*
void TableDelegate::StartCollectThread() {
  collect_thread_=std::thread([this] {CollectThread();});
}
void TableDelegate::CollectThread(){
  collect_flag_=true;
  while(collect_flag_){
    K key;
    V val;
    // may collect splits of other params used later
    if(table_->async_get_collect(&key,&val)){
      int splitid=key.key();
      //LOG(INFO)<<"collected "<<splitid;
      auto& split=split_param_map_.at(splitid);
      Param* p=split.first;
      int offset=split.second;
      float * dptr = p->mutable_data()->dptr();
      for(auto v: val.data().value())
        dptr[offset++]=v;
      // check this split is complete, i.e. offset is the start of next split
      if(split_param_map_.find(key.key()+1)!=split_param_map_.end())
        CHECK_EQ(offset, split_param_map_.at(key.key()+1).second);
      asyncget_split_[splitid]=true;
      collect_counter_[p->id()]+=1;
    }else{
      //std::this_thread::yield();
      sleep(0.0001);
    }
  }
}


void TableDelegate::Collect(Param * param, int step){
  auto& splits=param_splits_[param->id()];
  int nsplits=splits.size();
  while(true){
    int ncollect=collect_counter_[param->id()];
    CHECK_LE(ncollect, nsplits);
    if(ncollect==nsplits){
      for(auto& split:splits){
        CHECK(asyncget_split_[split.first]);
        asyncget_split_[split.first]=false;
      }
      collect_counter_[param->id()]=0;
      break;
    }
    else{
      sleep(0.0001);
      //std::this_thread::yield();
    }
  }
}

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

TypedGlobalTable<Tkey, TVal>* TableDelegate::CreateParamTable(){
  auto *info = new TableDescriptor(0, GlobalContext::Get()->num_servers());
  info->key_marshal = new Marshal<TKey>;
  info->value_marshal = new Marshal<TVal>;
  info->sharder = TKeySharder;
  // TODO remove accum
  info->accum = nullptr;
  info->partition_factory = new typename SparseTable<K, V>::Factory;
  auto table=new TypedGlobalTable<TKey, TVal>();
  table->Init(info);
  //LOG(INFO)<<"table shards num "<<table->num_shards();
  return table;
}

*/



}  // namespace lapis
