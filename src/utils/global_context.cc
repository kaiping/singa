// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-06-28 14:40
#include <glog/logging.h>
#include "utils/global_context.h"
#include "proto/system.pb.h"
#include "utils/proto_helper.h"

namespace lapis {

std::shared_ptr<GlobalContext> GlobalContext::instance_;
int GlobalContext::kCoordinator;
GlobalContext::GlobalContext(const std::string &system_conf):system_conf_(system_conf) {
  SystemProto proto;
  ReadProtoFromTextFile(system_conf.c_str(), &proto);
  standalone_=proto.standalone();
  synchronous_= proto.synchronous();
  table_server_start_=proto.cluster().server_start();
  table_server_end_=proto.cluster().server_end();;
  for(auto& group: proto.cluster().group()){
    vector<int> workers;
    for(int i=group.start();i<group.end();i++){
      workers.push_back(i);
    }
    groups_.push_back(workers);
  }
  shard_folder_=proto.cluster().shard_folder();
}

void GlobalContext::Setup(const std::shared_ptr<NetworkThread>& nt){
  rank_=nt->id();
  num_procs_=nt->size();
  kCoordinator=nt->size()-1;
  int gid=0, gid_=-1;
  for(auto& group: groups_){
    worker_id_=0;
    for(auto& worker: group){
      if(worker==rank_){
        gid_=gid;
        break;
      }else
        worker_id_++;
    }
    if(gid_!=-1)
      break;
    gid++;
  }
  CHECK(gid_==-1||rank_==kCoordinator);
  VLOG(3)<<"init network thread";
}
shared_ptr<GlobalContext> GlobalContext::Get(const std::string &sys_conf){
  if(!instance_) {
    instance_.reset(new GlobalContext(sys_conf));
    auto nt=NetworkThread::Get();
    instance_->Setup(nt);
  }
  return instance_;
}
void GlobalContext::Finish() {
  NetworkThread::Get()->Shutdown();
}
shared_ptr<GlobalContext> GlobalContext::Get() {
  if(!instance_) {
    LOG(ERROR)<<"The first call to Get should "
              <<"provide the sys/model conf path";
  }
  return instance_;
}
}  // namespace lapis
