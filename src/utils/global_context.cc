// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-06-28 14:40
#include <glog/logging.h>
#include "utils/global_context.h"
#include "proto/system.pb.h"
#include "utils/proto_helper.h"
#include "utils/network_thread.h"

namespace lapis {

std::shared_ptr<GlobalContext> GlobalContext::instance_;
int GlobalContext::kCoordinator;
GlobalContext::GlobalContext(const std::string &system_conf,)system_conf_(system_conf) {
  SystemProto proto;
  ReadProtoFromTextFile(system_conf.c_str(), &proto);
  standalone_=proto.standalone();
  synchronous_= proto.synchronous();
  table_server_start_=proto.cluster().server_start();
  table_server_end_=proto.cluster().server_end();;
  for(auto& group: proto.cluster().group()){
    vector<int> workers;
    for(int i=group.start();i<group.end();i++)
      workers.push_back(i);
    groups_.push_back(workers);
  }
  shard_folder_=proto.cluster().shard_folder();
}

shared_ptr<GlobalContext> GlobalContext::Get(const std::string &sys_conf){
  if(!instance_) {
    instance_.reset(new GlobalContext(sys_conf));
    auto nt=NetworkThread::Get();
    rank_=nt->id();
    num_procs_=nt->size();
    kCoordinator=net->size()-1;
    int gid=0, gid_=-1;
    for(auto& group: groups_){
      for(auto& worker: group)
        if(worker==rank_){
          gid_=gid;
          break;
        }
      gid++;
    }
    if(gid==-1)
      CHECK(rank_==kCoordinator);
    VLOG(3)<<"init network thread";
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
