// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-06-28 14:40

#include "utils/global_context.h"
#include "proto/system.pb.h"
#include "utils/proto_helper.h"
#include "utils/network_thread.h"

namespace lapis {

std::shared_ptr<GlobalContext> GlobalContext::instance_;

GlobalContext::GlobalContext(const std::string &system_conf,
    const std::string &model_conf): model_conf_(model_conf) {
  SystemProto proto;
  ReadProtoFromTextFile(system_conf.c_str(), &proto);
  standalone_=proto.standalone();
  synchronous_= proto.synchronous();
  if (proto.has_mem_server_start() && proto.has_mem_server_end()) {
    mem_server_start_=proto.mem_server_start();
    mem_server_end_=proto.mem_server_end();
  }
  else {
    mem_server_start_=0;
    mem_server_start_=0;
  }
}

shared_ptr<GlobalContext> GlobalContext::Get(const std::string &sys_conf,
                                             const std::string &model_conf) {
  if(!instance_) {
    instance_.reset(new GlobalContext(sys_conf, model_conf));
    VLOG(3)<<"init network thread";
    NetworkThread::Init();
    auto net=NetworkThread::Get();
    rank_=->id();
    num_processes_=net->size();
    if(mem_server_start_==0&&mem_server_end_==0){
      mem_server_start_=0;
      mem_server_end_=num_processes_;
    } else{
      CHECK(mem_server_start_>=0);
      CHECK(mem_server_start_<mem_server_end_);
      CHECK(mem_server_end_<=num_processes_);
    }
  }
  return instance_;
}

shared_ptr<GlobalContext> GlobalContext::Get() {
  if(!instance_) {
    LOG(ERROR)<<"The first call to Get should "
              <<"provide the sys/model conf path";
  }
  return instance_;
}
}  // namespace lapis
