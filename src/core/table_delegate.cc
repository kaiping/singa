// Copyright © 2014 Wei Wang & Jinyang Gao. All Rights Reserved.
// 2014-07-18 16:33

#include <vector>

#include "core/table_delegate.h"
#include "proto/model.pb.h"
#include "da/dary.h"
#include "utils/network_service.h"
#include "proto/worker.pb.h"

namespace lapis {

void TableDelegate::SplitParams(const vector<Param *>& params, int worker_id) {
	/*
  int total_splits=0;
  for(auto param: params){
    total_splits+=SplitParam(param, worker_id);
  }
  LOG(INFO)<<"Total splits for this worker "<<total_splits;
  */
}

int TableDelegate::SplitParam(Param * param, int worker_id){
/*
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
  CHECK_EQ(splits_.size(), param->id());
  splits_.push_back(splits);
  // for debug
  for(auto& split: splits){
    char tmpbuf[1024];
    sprintf(tmpbuf, "%4d %5d %5d", param->id(), split.first, split.second);
    LOG(INFO)<<string(tmpbuf);
  }
  return splits.size();
*/
	return 0;

}
void TableDelegate::Update(const std::vector<Param*> &params, int step) {
	/*
  for(auto* param: params)
    Update(param,step);
  return;
  */
}

void TableDelegate::Update(Param *param, int step){
	/*
  TKey key;
  key.set_version(step);
  int offset = 0;
  const float * dptr = param->grad().dptr();
  for(auto& entry: splits_[param->id()]) {
    TVal val;
    DAryProto* data=val.mutable_data();

    for(int k = 0; k < entry.second; k++){
      data->add_value(dptr[offset++]);
    }

    key.set_id(entry.first);
    TableData tuple;
    tuple.set_shard(Shard(entry.first,GlobalContext::Get()->num_servers()));
    tuple.set_source(GlobalContext::Get()->rank());
    tuple.set_table(0);
    Arg *arg=tuple.add_kv_data();
    std::string *keystr=arg->mutable_key();
    key.SerializeToString(keystr);
    std::string *valstr=arg->mutable_value();
    val.SerializeToString(valstr);
    tuple.set_done(true);
    NetworkService::Get()->Send(tuple.shard(), MTYPE_UPDATE_REQUEST, tuple);
  }
  */
}


/*
void TableDelegate::Get(const std::vector<Param*> &params, int step){
  for(auto* param : params)
    Get(param, step);
  return;
}

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
	/*
  for(auto* param : params)
    AsyncGet(param, step);
  return;
  */
}
void TableDelegate::Put(const std::vector<Param*> &params, int step) {
	/*
  for(auto* param: params)
    Put(param, step);
    */
}
void TableDelegate::Put(Param * param, int step){
	/*
  int offset = 0;
  const float * data_addr = param->data().dptr();
  for(auto& entry: splits_[param->id()]) {
    TVal val;
    // sgd related hyper-parameters
    val.set_learning_rate_multiplier(param->learning_rate_multiplier());
    val.set_weight_decay_multiplier(param->weight_decay_multiplier());
    val.set_param_id(param->id());
    val.set_split_id(entry.first);
    val.set_split_offset(split_map_[entry.first].second);
    DAryProto* dary=val.mutable_data();
    for(int k = 0; k < entry.second; k++){
      dary->add_value(data_addr[offset]);
      offset++;
    }
    TKey key;
    key.set_version(0);
    key.set_id(entry.first);
    TableData tuple;
    tuple.set_shard(Shard(entry.first,GlobalContext::Get()->num_servers()));
    tuple.set_source(GlobalContext::Get()->rank());
    tuple.set_table(0);
    Arg *arg=tuple.add_kv_data();
    std::string *keystr=arg->mutable_key();
    key.SerializeToString(keystr);
    std::string *valstr=arg->mutable_value();
    val.SerializeToString(valstr);
    tuple.set_done(true);

    NetworkService::Get()->Send(tuple.shard(), MTYPE_UPDATE_REQUEST, tuple);
    //Network::Get()->Send(tuple.shard(), MTYPE_UPDATE_REQUEST, tuple);
  }
  */
}

void TableDelegate::AsyncGet(Param * param, int step){
	/*
  TKey key;
  key.set_version(step);
  for(auto entry: splits_[param->id()]) {
    key.set_id(entry.first);
    HashGet req;
    std::string *keystr=req.mutable_key();
    key.SerializeToString(keystr);
    req.set_table(0);
    req.set_shard(Shard(entry.first, GlobalContext::Get()->num_servers()));
    req.set_source(GlobalContext::Get()->rank());
    NetworkService::Get()->Send(req.shard(), MTYPE_GET_REQUEST, req);
    asyncget_split_.at(entry.first)=false;
    //LOG(INFO)<<"get "<<entry.first;
  }
  */
}
void TableDelegate::AsyncCollect(Param * param, int step){
	/*
  auto& splits=splits_[param->id()];
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
      if(split_map_.find(key.id()+1)!=split_map_.end())
        CHECK_EQ(offset, split_map_.at(key.id()+1).second);
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
  */
}

}  // namespace lapis
