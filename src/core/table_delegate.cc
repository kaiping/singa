#include <vector>

#include "core/table_delegate.h"
#include "proto/model.pb.h"
#include "da/dary.h"
#include "utils/network_service.h"

namespace lapis {
const int kMaxParamLen=1<30;

void TableDelegate::Setup(const vector<Param*>& params,
    TableServerHandler* handler){
  // no table servers
  if(GlobalContext::Get()->num_servers()==0){
    for(auto param: params){
      // one tuple per param
      local_tuple_[param->id()];
    }
    CHECK(handler!=nullptr);
    handler_=handler;
  }
  SplitParams(params, GlobalContext::Get()->worker_id());
}

void TableDelegate::SplitParams(const vector<Param *>& params, int worker_id) {
  int total_splits=0;
  for(auto param: params){
    total_splits+=SplitParam(param, worker_id);
  }
  LOG(INFO)<<"Total splits for this worker "<<total_splits;
}

int TableDelegate::SplitParam(Param * param, int worker_id){
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
  // do not split if there is no table server, updates will be done locally
  if(GlobalContext::Get()->num_table_servers()==0){
    CHECK_LE(local_size, kMaxParamLen);
    splitsize=kMaxParamLen;
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
  splits_.push_back(splits);
  // for debug
  for(auto& split: splits){
    char tmpbuf[1024];
    sprintf(tmpbuf, "%4d %5d %5d", param->id(), split.first, split.second);
    LOG(INFO)<<string(tmpbuf);
  }
  return splits.size();
}

void TableDelegate::Update(const std::vector<Param*> &params, int step) {
  for(auto* param: params)
    Update(param,step);
  return;
}

void TableDelegate::Update(Param *param, int step){
 int offset = 0;
  const float * dptr = param->grad().dptr();
  for(auto& entry: splits_[param->id()]) {
    TVal tval;
    DAryProto* grad=tval.mutable_grad();
    for(int k = 0; k < entry.second; k++){
      grad->add_value(dptr[offset++]);
    }
    if(GlobalContext::Get()->num_table_servers()==0){
      CHECK_EQ(splits_[param->id()].size(), 1);
      handler_->Update(&local_tuple_[entry.first], tval);
    } else{
      RequestBase request;
      request.set_table(0);
      request.set_source(NetworkService::Get()->id());
      UpdateRequest *update_req = request.MutableExtension(UpdateRequest::name);
      int shard= Sharding(entry.first, GlobalContext::Get()->num_servers());
      update_req->set_shard(shard);
      TableData *tuple=update_req->mutable_data();
      TKey* key=tuple->mutable_key();
      key->set_version(step);
      key->set_id(entry.first);
      tuple->set_allocated_value(&tval);
      NetworkService::Get()->Send(shard, MTYPE_REQUEST, request);
    }
  }
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
void TableDelegate::Put(const std::vector<Param*> &params, int step) {
  for(auto* param: params)
    Put(param, step);
}
void TableDelegate::Put(Param * param, int step){
  int offset=0;
  const float * data_addr = param->data().dptr();
  for(auto& entry: splits_[param->id()]) {
    TKey key;
    key.set_id(entry.first);
    key.set_version(step);

    TVal tval;
    // sgd related hyper-parameters
    tval.set_learning_rate_multiplier(param->learning_rate_multiplier());
    tval.set_weight_decay_multiplier(param->weight_decay_multiplier());
    tval.set_param_id(param->id());
    tval.set_split_id(entry.first);
    tval.set_split_offset(split_map_[entry.first].second);
    DAryProto* dary=tval.mutable_data();
    for(int k = 0; k < entry.second; k++){
      dary->add_value(data_addr[offset]);
      offset++;
    }
    if(GlobalContext::Get()->num_servers()==0){
      tval.set_threshold(1);
      handler_->Put(key, &local_tuple_[entry.first], tval);
    }else{
      RequestBase request;
      request.set_table(0);
      request.set_source(NetworkService::Get()->id());
      PutRequest *put_req = request.MutableExtension(PutRequest::name);
      int shard=Sharding(entry.first, GlobalContext::Get()->num_servers());
      put_req->set_shard(shard);
      TableData* tuple=put_req->mutable_data();
      if(GlobalContext::Get()->synchronous())
        tval.set_threshold(GlobalContext::Get()->num_groups());
      else
        tval.set_threshold(1);
      tuple->set_allocated_key(&key);
      tuple->set_allocated_value(&tval);
      NetworkService::Get()->Send(shard, MTYPE_REQUEST, request);
    }
  }
}
void TableDelegate::AsyncGet(const std::vector<Param*> &params, int step){
  for(auto* param : params)
    AsyncGet(param, step);
  return;
}

void TableDelegate::AsyncGet(Param * param, int step){
  if(GlobalContext::Get()->num_table_servers()==0)
    return;
  for(auto entry: splits_[param->id()]) {
    RequestBase request;
    request.set_table(0);
    request.set_source(NetworkService::Get()->id());
    GetRequest *get_req = request.MutableExtension(GetRequest::name);
    int shard= Sharding(entry.first, GlobalContext::Get()->num_servers());
    get_req->set_shard(shard);
    TKey *key = get_req->mutable_key();
    key->set_id(entry.first);
    key->set_version(step);
    NetworkService::Get()->Send(shard, MTYPE_REQUEST, request);
    asyncget_split_.at(entry.first)=false;
  }
}
void TableDelegate::AsyncCollect(Param * param, int step){
  auto& splits=splits_[param->id()];
  unsigned int nget=0;
  int start_split_id=splits.front().first;
  int end_split_id=splits.back().first;
  // check num of splits collected before
  for(auto& split: splits){
    if(asyncget_split_.at(split.first))
      nget++;
  }
  while(nget<splits.size()){
    // may collect splits of other params used later
    TVal *tval;
    int splitid;
    if(GlobalContext::Get()->num_servers()==0){
      CHECK_EQ(start_split_id, end_split_id);
      tval=&local_tuple_[start_split_id];
      splitid=start_split_id;
    } else{
      Message* resp=NetworkService::Get()->Receive();
      if(resp!=nullptr){
        // todo parse the resp msg to get TVal
        splitid=0;
        tval=NULL;
      }
    }
    if(tval!=NULL){
      auto& split=split_map_.at(splitid);
      Param* p=split.first;
      int offset=split.second;
      float * dptr = p->mutable_data()->dptr();
      for(auto v: tval->data().value())
        dptr[offset++]=v;
      // check this split is complete, i.e. offset is the start of next split
      if(split_map_.find(splitid+1)!=split_map_.end())
        CHECK_EQ(offset, split_map_.at(splitid+1).second);
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

}  // namespace lapis
