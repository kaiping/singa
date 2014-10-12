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
GlobalContext::GlobalContext(const std::string &system_conf,
    const std::string &model_conf): model_conf_(model_conf), system_conf_(system_conf) {
  SystemProto proto;
  ReadProtoFromTextFile(system_conf.c_str(), &proto);
  standalone_=proto.standalone();
  synchronous_= proto.synchronous();
  table_server_start_=proto.cluster().server_start();
  table_server_end_=proto.cluster().server_end();;
}
void GlobalContext::set_num_processes(int num){
  num_processes_=num;
  CHECK_GE(num,2)<<"must start at least 2 processes";
  if(table_server_end_==0)
    table_server_end_=num-1;
  else
    CHECK_LE(table_server_end_, num-1);
}

shared_ptr<GlobalContext> GlobalContext::Get(const std::string &sys_conf,
    const std::string &model_conf) {
  if(!instance_) {
    instance_.reset(new GlobalContext(sys_conf, model_conf));
    auto net=NetworkThread::Get();
    instance_->set_rank(net->id());
    instance_->set_num_processes(net->size());
    kCoordinator=net->size()-1;
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
