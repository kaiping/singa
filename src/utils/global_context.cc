// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-06-28 14:40
#include <glog/logging.h>
#include "utils/global_context.h"
#include "proto/system.pb.h"
#include "utils/proto_helper.h"
#include "utils/common.h"


namespace lapis {

std::shared_ptr<GlobalContext> GlobalContext::instance_;
int GlobalContext::kCoordinator;
GlobalContext::GlobalContext(const std::string &system_conf):system_conf_(system_conf) {
  SystemProto proto;
  ReadProtoFromTextFile(system_conf.c_str(), &proto);
  standalone_=proto.standalone();
  synchronous_= proto.synchronous();
  table_server_start_=proto.cluster().server_start();
  table_server_end_=proto.cluster().server_end();
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
  mpi_=nt;
  rank_=nt->id();
  num_procs_=nt->size();
  kCoordinator=nt->size()-1;
  int gid=0;
  gid_=-1;
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
  LOG(INFO)<<"GlobalConxt Setup finish: "<<
    "Group id "<<gid_<<" rank "<<rank_<<" group rank "<<worker_id_;
  CHECK(gid_!=-1||rank_==kCoordinator||AmITableServer());
  if(gid_!=-1) {
    int gsize=group_size();
    int *member=new int[gsize];
    int k=0;
    for(auto mem: groups_[gid_])
      member[k++]=mem;
    MPI_Group world_group;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);
    MPI_Group_incl(world_group, gsize, member, &mpigroup_);
    MPI_Comm_create_group(MPI_COMM_WORLD, mpigroup_,0, &mpicomm_);
    delete member;
  }
}
shared_ptr<GlobalContext> GlobalContext::Get(const std::string &sys_conf){
  if(!instance_) {
    instance_.reset(new GlobalContext(sys_conf));
    instance_->Setup(NetworkThread::Get());
  }
  return instance_;
}
void GlobalContext::Shutdown() {
  mpi_->Shutdown();
  if(gid_!=-1){
    MPI_Group_free(&mpigroup_);
    MPI_Comm_free(&mpicomm_);
  }
  MPI_Finalize();
}
shared_ptr<GlobalContext> GlobalContext::Get() {
  if(!instance_) {
    LOG(ERROR)<<"The first call to Get should "
              <<"provide the sys/model conf path";
  }
  return instance_;
}
}  // namespace lapis
