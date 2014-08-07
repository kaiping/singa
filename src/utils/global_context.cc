// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-06-28 14:40
#include <glog/logging.h>
#include "utils/global_context.h"
#include "proto/system.pb.h"
#include "utils/proto_helper.h"
#include "utils/network_thread.h"

namespace lapis {

std::shared_ptr<GlobalContext> GlobalContext::instance_;

const int GlobalContext::kCoordinatorRank=0;

GlobalContext::GlobalContext(const std::string &system_conf,
    const std::string &model_conf): model_conf_(model_conf) {
  auto net=NetworkThread::Get();
  rank_=net->id();
  num_processes_=net->size();

  SystemProto proto;
  ReadProtoFromTextFile(system_conf.c_str(), &proto);
  standalone_=proto.standalone();
  synchronous_= proto.synchronous();
  if (proto.has_table_server_start() && proto.has_table_server_end()) {
    table_server_start_=proto.table_server_start();
    table_server_end_=proto.table_server_end();
    CHECK(table_server_start_>=0);
    CHECK(table_server_start_<table_server_end_);
    CHECK(table_server_end_<=num_processes_);
  }
  else {
    table_server_start_=0;
    table_server_end_=num_processes_;
  }
}

shared_ptr<GlobalContext> GlobalContext::Get(const std::string &sys_conf,
    const std::string &model_conf) {
  if(!instance_) {
    instance_.reset(new GlobalContext(sys_conf, model_conf));
    VLOG(3)<<"init network thread";
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
