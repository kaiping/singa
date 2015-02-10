#include <vector>
#include "core/table_delegate.h"
#include "proto/model.pb.h"
#include "utils/network.h"
namespace singa {
const size_t kMaxParamLen=1<<30;
// max num of splits per parameter, used to create split id
// 3571 is the 500-th prime number
const int kMaxSplits=3571;

Delegate::Delegate(int worker_id, int rank, int num_servers,
    int group_size, int num_groups, bool synchronous):
    worker_id_(worker_id), rank_(rank), num_servers_(num_servers),
    group_size_(group_size), num_groups_(num_groups), synchronous_(synchronous){
  Start();
}
Delegate::Delegate(shared_ptr<Cluster> cluster){
  worker_id_=cluster->worker_id();
  rank_=cluster->rank();
  num_servers_=cluster->num_servers();
  num_groups_=cluster->num_groups();
  group_size_=cluster->group_size();
  largest_message_=cluster->largest_message();
  synchronous_=cluster->synchronous();
  Start();
}

Delegate::~Delegate(){
  /*
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
  */
}

void Delegate::Start(){
  LOG(INFO)<<"Start Delegate with "<<num_servers_<<" table servers";
  running_=true;
  //running_loop_=new std::thread(&Delegate::InternalThread, this);
}

void Delegate::Setup(const vector<Param *>& params){
  int splitid=0, total_size=0;
  for(Param* param: params){
    int size=param->data().count()*sizeof(float);
    total_size+=size;
    int nsplits=size/largest_message_+(size%largest_message_!=0);
    DLOG(INFO)<<"Parameter "<<param->id()<<" has "<<nsplits<<" splits";
    vector<shared_ptr<Split>> splits;
    for(auto j = 0, pos=0; j < nsplits; j++) {
      int len=(pos+largest_message_)<size?largest_message_:size-pos;
      splits.push_back(make_shared<Split>(splitid++, pos, len, param));
      id_to_split_[splitid]=splits.back();
      pos+=len;
    }
    paramid_to_splits_[param->id()]=splits;
  }
  //sharder setup..
}

void Delegate::Update(const std::vector<Param*> &params, int step) {
  if(paramid_to_splits_.size())
    Setup(params);
  for(auto* param: params)
    Update(param,step);
  return;
}

void Delegate::Update(Param *param, int step){
  CHECK(paramid_to_splits_.find(param->id())==paramid_to_splits_.end());
  for(auto& entry: paramid_to_splits_[param->id()]) {
    RequestBase request;
    request->set_table(0);
    request->set_source(rank_);
    request->add_destination(Sharding(entry->id, num_servers_));
    UpdateRequest *update_req = request->MutableExtension(UpdateRequest::name);
    TableData *tuple=update_req->mutable_data();
    TKey* key=tuple->mutable_key();
    key->set_version(step);
    key->set_id(entry->id);
    string *val=tuple->mutable_value();
    CHECK_EQ(param, entry->param);
    param->SerializeTo(val, entry->offset, entry->len);
    NetworkService::Get()->Send(MTYPE_REQUEST,request);
  }
}

void Delegate::Put(const std::vector<Param*> &params, int step) {
  if(paramid_to_splits_.size()==0)
    Setup(params);
  for(auto* param: params)
    Put(param, step);
}

void Delegate::Put(Param * param, int step){
  CHECK(paramid_to_splits_.find(param->id())==paramid_to_splits_.end());
  for(auto& entry: paramid_to_splits_[param->id()]) {
    // prepare put request
    RequestBase request;
    request->set_table(0);
    request->set_source(rank_);
    request->add_destination(Sharding(entry->id,num_servers_));
    PutRequest *put_req = request->MutableExtension(PutRequest::name);
    TableData* tuple=put_req->mutable_data();
    tuple->set_key(entry->id());
    tuple->set_version(step);
    string* val=tuple->mutable_value();
    param->SerializeTo(val, entry->offset, entry->len);
    if(synchronous_)
      tuple->set_threshold(num_groups_);
    else
      tuple->set_threshold(1);
    NetworkService::Get()->Send(MTYPE_REQUEST, request);
  }
}

void Delegate::AsyncGet(const std::vector<Param*> &params, int step){
  if(paramid_to_splits_.size()==0)
    Setup(params);
  for(auto* param : params)
    AsyncGet(param, step);
  return;
}

void Delegate::AsyncGet(Param * param, int step){
  for(auto entry: paramid_to_splits_[param->id()]) {
    shared_ptr<RequestBase> request(new RequestBase());
    request->set_table(0);
    request->set_source(rank_);
    request->add_destination(Sharding(entry->id, num_servers_));
    GetRequest *get_req = request->MutableExtension(GetRequest::name);
    TableData* tuple=get_req->mutable_data();
    tuple->set_key(entry->id());
    tuple->set_version(step);
    NetworkService::Get()->Send(MTYPE_REQUEST, request);
    if(split_collected_.find(entry->id)==split_collected_.end())
      split_collected_[entry->id]=false;
    else
      CHECK(!split_collected_[entry->id]);
  }
}

void Delegate::AsyncCollect(Param * param, int step){
  unsigned int nget=0;
  auto& splits=paramid_to_splits_[param->id()];
  while(nget<splits.size()){
    // check num of splits collected before
    for(auto& split: splits){
      if(!split_collected_.at(split->id)){
        //split_collected_[split->id]=false;
        shared_ptr<TableData> resp=
          NetworkService::Get()->TryRecv(split->id(), MTYPE_RESPONSE);
        if(resp==nullptr)
          sleep(FLAGS_sleep_time);
        else{
          split_collected_.at(split->id)=true;
          nget++;
          CHECK_EQ(resp->key(), split->id);
          param->ParseFrom(resp->value(), split->offset, split->len);
        }
      }
    }
  }
  for(auto& split: splits)
      split_collected_.at(split->id)=false;
  //LOG(ERROR)<<"Get param splits "<<param->id()<<" rank "<<Cluster::Get()->rank();
}

/*
void Delegate::InternalThread(){
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
        LOG(FATAL)<<"Delegate receieved unexpected message (code) "<<tag;
      }
    }
    if(sleeping)
      sleep(0.0001);
  }
}
*/
}  // namespace singa
