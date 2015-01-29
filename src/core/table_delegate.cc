#include <vector>
#include "core/table_delegate.h"
#include "proto/model.pb.h"
#include "utils/network.h"
namespace singa {
const size_t kMaxParamLen=1<<30;
// max num of splits per parameter, used to create split id
// 3571 is the 500-th prime number
const int kMaxSplits=3571;

TableDelegate::TableDelegate(int worker_id, int rank, int num_servers,
    int group_size, int num_groups, bool synchronous,
    std::shared_ptr<TableServerHandler> handler):
    worker_id_(worker_id), rank_(rank), num_servers_(num_servers),
    group_size_(group_size), num_groups_(num_groups), synchronous_(synchronous),
    handler_(handler){
  Start();
}
TableDelegate::TableDelegate(shared_ptr<Cluster> cluster,
    std::shared_ptr<TableServerHandler> handler){
  worker_id_=cluster->worker_id();
  rank_=cluster->rank();
  num_servers_=cluster->num_servers();
  num_groups_=cluster->num_groups();
  group_size_=cluster->group_size();
  synchronous_=cluster->synchronous();
  handler_=handler;
  Start();
}

TableDelegate::~TableDelegate(){
  auto cluster=Cluster::Get();
  if(cluster->num_servers()>0){
    // ensure the sending queue is empty.
    while(true){
      if(sending_queue_.size()>0){
        sleep(0.0001);
      }else{
        break;
      }
    }
    MPI_Barrier(cluster->worker_comm());
    if(cluster->rank()==cluster->worker_start()&&cluster->num_servers()){
      string msg;
      for (int i=cluster->server_start();i<cluster->server_end(); i++){
        Network::Get()->Send(i, MTYPE_SHUTDOWN,msg);
      }
    }
    //DLOG(ERROR)<<"wait";
    running_loop_->join();
    //DLOG(ERROR)<<"joined";
    delete running_loop_;
  }
}

void TableDelegate::Start(){
  if(num_servers_==0){
    LOG(INFO)<<"Start TableDelegate without table servers, do updates locally";
    CHECK(handler_!=nullptr)<<"Must provide TableServerHandler";
  }else{
    LOG(INFO)<<"Start TableDelegate with "<<num_servers_<<" table servers";
    running_=true;
    running_loop_=new std::thread(&TableDelegate::InternalThread, this);
  }
}

void TableDelegate::SplitParam(Param * param){
  int id=param->id()*group_size_;
  if(param->partition()>=0)
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
  DLOG(INFO)<<"Parameter "<<param->id()<<" has "<<nsplits<<" splits";
  CHECK_LE(nsplits,kMaxSplits)<<"total splits for one param partition "
    <<" exceeds kMaxSplits, raise kMaxSplits in solver config";
  vector<shared_ptr<Split>> splits;
  for(auto j = 0, pos=0; j < nsplits; j++) {
    int len=(pos+splitsize)<local_size?splitsize:local_size-pos;
    int splitid=id*kMaxSplits+j;
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
    TVal *tval=new TVal();
    DAryProto* grad=tval->mutable_grad();
    for(int k = 0; k < entry->len; k++){
      grad->add_value(dptr[offset++]);
    }
    if(num_servers_==0){
      // local update
      CHECK_EQ(paramid_to_splits_[param->id()].size(), 1);
      handler_->Update(&local_tuple_[entry->id], *tval);
      delete tval;
    } else{
      // prepare update request.
      shared_ptr<RequestBase> request(new RequestBase());
      request->set_table(0);
      request->set_source(rank_);
      request->set_shard(Sharding(entry->id, num_servers_));
      UpdateRequest *update_req = request->MutableExtension(UpdateRequest::name);
      TableData *tuple=update_req->mutable_data();
      TKey* key=tuple->mutable_key();
      key->set_version(step);
      key->set_id(entry->id);
      tval->set_version(step);
      tuple->set_allocated_value(tval);
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
    TKey* key=new TKey();
    key->set_id(entry->id);
    key->set_version(step);

    TVal* tval=new TVal();
    // sgd related hyper-parameters
    tval->set_learning_rate_multiplier(param->learning_rate_multiplier());
    tval->set_weight_decay_multiplier(param->weight_decay_multiplier());
    tval->set_param_id(param->id());
    tval->set_split_id(entry->id);
    tval->set_split_offset(entry->offset);
    tval->set_version(step);
    DAryProto* dary=tval->mutable_data();
    for(int k = 0; k < entry->len; k++){
      dary->add_value(data_addr[offset]);
      offset++;
    }
    if(num_servers_==0){
      // local put
      tval->set_threshold(1);
      handler_->Put(*key, &local_tuple_[entry->id], *tval);
      delete key;
      delete tval;
    }else{
      // prepare put request
      shared_ptr<RequestBase> request(new RequestBase());
      request->set_table(0);
      request->set_source(rank_);
      request->set_shard(Sharding(entry->id,num_servers_));
      PutRequest *put_req = request->MutableExtension(PutRequest::name);
      TableData* tuple=put_req->mutable_data();
      if(synchronous_)
        tval->set_threshold(num_groups_);
      else
        tval->set_threshold(1);
      tuple->set_allocated_key(key);
      tuple->set_allocated_value(tval);
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
    shared_ptr<RequestBase> request(new RequestBase());
    request->set_table(0);
    request->set_source(rank_);
    request->set_shard(Sharding(entry->id, num_servers_));
    GetRequest *get_req = request->MutableExtension(GetRequest::name);
    TKey *key = get_req->mutable_key();
    key->set_id(entry->id);
    key->set_version(step);
    sending_queue_.push(request);
    if(split_collected_.find(entry->id)==split_collected_.end())
      split_collected_[entry->id]=false;
    else
      CHECK(!split_collected_[entry->id]);
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
      if(split_collected_.at(split->id)){
        nget++;
        split_collected_[split->id]=false;
      }
    }
    if(nget<splits.size())
      sleep(0.0001);
    //LOG(ERROR)<<"Get param splits "<<param->id()<<" rank "<<Cluster::Get()->rank();
  }
}

void TableDelegate::InternalThread(){
  int src, tag;
  while(true){
    string msg;
    bool sleeping=true;
    // sending
    if(sending_queue_.size()>0){
      shared_ptr<RequestBase> request=sending_queue_.front();
      request->SerializeToString(&msg);
      Network::Get()->Send(request->shard(), MTYPE_REQUEST,msg);
      sending_queue_.pop();
      sleeping=false;
    }
    if(Network::Get()->Recv(&tag, &src, &msg)){
      if(tag==MTYPE_SHUTDOWN){
        break;
      }else if(tag==MTYPE_RESPONSE){
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
        split_collected_.at(key.id())=true;
        sleeping=false;
      }else{
        LOG(FATAL)<<"TableDelegate receieved unexpected message (code) "<<tag;
      }
    }
    if(sleeping)
      sleep(0.0001);
  }
}
}  // namespace singa
