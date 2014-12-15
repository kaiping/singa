#include <vector>
#include <thread>
#include "core/table_delegate.h"
#include "proto/model.pb.h"
#include "utils/network.h"
namespace lapis {
const int kMaxParamLen=1<30;

TableDelegate::TableDelegate(int worker_id, int rank, int num_servers,
    int group_size, int num_groups, bool synchronous,
    TableServerHandler* handler):
    worker_id_(worker_id), rank_(rank), num_servers_(num_servers),
    group_size_(group_size), num_groups_(num_groups), synchronous_(synchronous),
    handler_(handler){
  Start();
}
TableDelegate::TableDelegate(shared_ptr<GlobalContext> gc,
    TableServerHandler* handler){
  worker_id_=gc->worker_id();
  rank_=gc->rank();
  num_servers_=gc->num_servers();
  num_groups_=gc->num_groups();
  group_size_=gc->group_size();
  synchronous_=gc->synchronous();
  handler_=handler;
  Start();
}

void TableDelegate::Start(){
  if(num_servers_==0){
    LOG(INFO)<<"Start TableDelegate without table servers, do updates locally";
    CHECK(handler_!=nullptr)<<"Must provide TableServerHandler";
  }else{
    LOG(INFO)<<"Start TableDelegate with "<<num_servers_<<" table servers";
    std::thread running_loop(&TableDelegate::InternalThread, this);
    running_loop.join();
  }
}

void TableDelegate::SplitParam(Param * param){
  int id=param->id()*group_size_;
  if(param->partition())
    id+=worker_id_;
  int splitsize=param->split_threshold();
  if(splitsize>=16777216){
    LOG(WARNING)<<"split of size "<<splitsize
      <<"  exceeds the size of max google protobuf message, i.e., 64MB"
      <<", reset the split threshold to 4000,000 Bytes";
    splitsize = 1000000;
  }
  // do not split if there is no table server, updates will be done locally
  int local_size=param->local_size();
  if(num_servers_==0){
    CHECK_LE(local_size, kMaxParamLen);
    splitsize=kMaxParamLen;
  }
  int nsplits=local_size/splitsize+(local_size%splitsize!=0);
  CHECK_LE(nsplits,kMaxSplits_)<<"total splits for one param partition "
    <<" exceeds kMaxSplits, raise kMaxSplits in solver config";
  vector<shared_ptr<Split>> splits;
  for(auto j = 0, pos=0; j < nsplits; j++) {
    int len=(pos+splitsize)<local_size?splitsize:local_size-pos;
    int splitid=id*kMaxSplits_+j;
    splits.push_back(make_shared<Split>(splitid, pos, len, param));
    id_to_split_[splitid]=splits.back();
    pos+=len;
  }
  paramid_to_splits_[param->id()]=splits;
}

void TableDelegate::Update(const std::vector<Param*> &params, int step) {
  for(auto* param: params)
    Update(param,step);
  return;
}

void TableDelegate::Update(Param *param, int step){
  int offset = 0;
  const float * dptr = param->gptr();
  if(paramid_to_splits_.find(param->id())==paramid_to_splits_.end())
    SplitParam(param);
  for(auto& entry: paramid_to_splits_[param->id()]) {
    TVal tval;
    DAryProto* grad=tval.mutable_grad();
    for(int k = 0; k < entry->len; k++){
      grad->add_value(dptr[offset++]);
    }
    if(num_servers_==0){
      // local update
      CHECK_EQ(paramid_to_splits_[param->id()].size(), 1);
      handler_->Update(&local_tuple_[entry->id], tval);
    } else{
      // prepare update request.
      shared_ptr<RequestBase> request=make_shared<RequestBase>();
      request->set_table(0);
      request->set_source(rank_);
      request->set_shard(Sharding(entry->id, num_servers_));
      UpdateRequest *update_req = request->MutableExtension(UpdateRequest::name);
      TableData *tuple=update_req->mutable_data();
      TKey* key=tuple->mutable_key();
      key->set_version(step);
      key->set_id(entry->id);
      tuple->set_allocated_value(&tval);
      sending_queue_.push(request);
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
    key.set_id(entry->id);
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
  if(paramid_to_splits_.find(param->id())==paramid_to_splits_.end())
    SplitParam(param);

  int offset=0;
  const float * data_addr = param->data().dptr();
  for(auto& entry: paramid_to_splits_[param->id()]) {
    TKey key;
    key.set_id(entry->id);
    key.set_version(step);

    TVal tval;
    // sgd related hyper-parameters
    tval.set_learning_rate_multiplier(param->learning_rate_multiplier());
    tval.set_weight_decay_multiplier(param->weight_decay_multiplier());
    tval.set_param_id(param->id());
    tval.set_split_id(entry->id);
    tval.set_split_offset(entry->offset);
    DAryProto* dary=tval.mutable_data();
    for(int k = 0; k < entry->len; k++){
      dary->add_value(data_addr[offset]);
      offset++;
    }
    if(num_servers_==0){
      // local put
      tval.set_threshold(1);
      handler_->Put(key, &local_tuple_[entry->id], tval);
    }else{
      // prepare put request
      shared_ptr<RequestBase> request=make_shared<RequestBase>();
      request->set_table(0);
      request->set_source(rank_);
      request->set_shard(Sharding(entry->id,num_servers_));
      PutRequest *put_req = request->MutableExtension(PutRequest::name);
      TableData* tuple=put_req->mutable_data();
      if(synchronous_)
        tval.set_threshold(num_groups_);
      else
        tval.set_threshold(1);
      tuple->set_allocated_key(&key);
      tuple->set_allocated_value(&tval);
      sending_queue_.push(request);
    }
  }
}
void TableDelegate::AsyncGet(const std::vector<Param*> &params, int step){
  for(auto* param : params)
    AsyncGet(param, step);
  return;
}

void TableDelegate::AsyncGet(Param * param, int step){
  // no table servers->do not send request.
  if(num_servers_==0)
    return;
  if(paramid_to_splits_.find(param->id())==paramid_to_splits_.end())
    SplitParam(param);
  for(auto entry: paramid_to_splits_[param->id()]) {
    shared_ptr<RequestBase> request=make_shared<RequestBase>();
    request->set_table(0);
    request->set_source(rank_);
    request->set_shard(Sharding(entry->id, num_servers_));
    GetRequest *get_req = request->MutableExtension(GetRequest::name);
    TKey *key = get_req->mutable_key();
    key->set_id(entry->id);
    key->set_version(step);
    sending_queue_.push(request);
    split_collected_.at(entry->id)=false;
  }
}
void TableDelegate::AsyncCollect(Param * param, int step){
  if(num_servers_==0){
    auto& splits=paramid_to_splits_[param->id()];
    CHECK_EQ(splits.size(),1);
    float * dptr = param->mutable_dptr();
    int offset=0;
    for(auto v: local_tuple_[splits[0]->id].data().value())
      dptr[offset++]=v;
    return;
  }

  auto& splits=paramid_to_splits_[param->id()];
  unsigned int nget=0;
  while(nget<splits.size()){
    // check num of splits collected before
    for(auto& split: splits){
      if(split_collected_.at(split->id))
        nget++;
    }
    if(nget<splits.size())
      sleep(0.0001);
    nget=0;
  }
}

void TableDelegate::InternalThread(){
  bool sleeping=true;
  string msg;
  int src, tag;
  while(running_){
    // sending
    shared_ptr<RequestBase> request;
    sending_queue_.pop(&request);
    if(request){
      request->SerializeToString(&msg);
      Network::Get()->Send(request->shard(), MTYPE_REQUEST,msg);
      sleeping=false;
    }
    if(Network::Get()->Recv(&tag, &src, &msg)){
      TableData tuple;
      tuple.ParseFromString(msg);
      const TVal& tval=tuple.value();
      const TKey& key=tuple.key();
      auto& split=id_to_split_[key.id()];
      Param* p=split->param;
      int offset=split->offset;
      CHECK_EQ(split->len, tval.data().value_size());
      float * dptr = p->mutable_data()->dptr();
      for(auto v: tval.data().value())
        dptr[offset++]=v;
      split_collected_[key.id()]=true;
      sleeping=false;
    }
    if(sleeping)
      sleep(0.0001);
  }
}
}  // namespace lapis
