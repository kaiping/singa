#include <glog/logging.h>
#include "utils/global_context.h"
#include "utils/network.h"
#include "proto/cluster.pb.h"
#include "proto/worker.pb.h"

namespace singa {

std::shared_ptr<GlobalContext> GlobalContext::instance_;
int GlobalContext::kCoordinator;
GlobalContext::GlobalContext(const Cluster &cluster) {
	cluster_ = cluster;
  char tmp[256];
  int len;
  MPI_Get_processor_name(tmp, &len);
  hostname_=string(tmp);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs_);
  kCoordinator=num_procs_-1;
  gid_=id_=-1;
  const int start=cluster.worker_start();
  const int end=cluster.worker_end();
  CHECK_LT(start, end);
  if(cluster.has_server_end()){
    CHECK_LT(cluster.server_start(), cluster.server_end())
      <<"server end should be smaller than server start";
    CHECK_LE(cluster.server_end(), start)
      <<"cluster ranks should be before worker ranks";
  }
  CHECK(cluster.group_size());
  for(int k=start;k<end;k++){
    int gid=(k-start)/cluster.group_size();
    int id=(k-start)%cluster.group_size();
    if(id==0)
      groups_.push_back({});
    if(k==rank_){
      gid_=gid;
      id_=id;
    }
    groups_.back().push_back(k);
  }
  CHECK(gid_!=-1||rank_==kCoordinator||AmITableServer())
    <<"gid "<<gid_<<" rank "<<rank_<< "start " << start << " end " << end << " istableserver "<<AmITableServer();
  if(gid_!=-1) {
    // setup worker's mpi group to be used in Barrier
    int *member=new int[cluster.group_size()];
    int k=0;
    for(auto rank: groups_[gid_])
      member[k++]=rank;
    MPI_Group world_group;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);
    MPI_Group_incl(world_group, cluster.group_size(), member, &mpigroup_);
    MPI_Comm_create_group(MPI_COMM_WORLD, mpigroup_,0, &mpicomm_);
    delete member;

    // setup the group containing all workers
    int *allworkers = new int[end-start];
    for (int i=start; i<end; i++)
      allworkers[i-start]=i;
    MPI_Group_incl(world_group, end-start, allworkers, &allworkers_);
    MPI_Comm_create_group(MPI_COMM_WORLD, allworkers_,0, &allworkers_comm_);
    delete allworkers;
  }
  LOG(ERROR)<<"GlobalContext Setup: "<<
    "Group id "<<gid_<<" rank "<<rank_<<" id within group "<<id_;
}

shared_ptr<GlobalContext> GlobalContext::Get(const Cluster& cluster){
  if(!instance_) {
    instance_.reset(new GlobalContext(cluster));
  }
  return instance_;
}
void GlobalContext::Finalize() {
  if(gid_!=-1){
    MPI_Barrier(allworkers_comm_);
    if(rank_==cluster_.worker_start()&&cluster_.has_server_end()){
      EmptyMessage msg;
      for (int i=cluster_.server_start(); i<cluster_.server_end(); i++){
        EmptyMessage msg;
        Network::Get()->Send(i, MTYPE_SHUTDOWN,msg);
      }
    }
    MPI_Comm_free(&mpicomm_);
    MPI_Group_free(&mpigroup_);
    MPI_Comm_free(&allworkers_comm_);
    MPI_Group_free(&allworkers_);
  }else{
    LOG(ERROR)<<"server shuting down";
  }
}
shared_ptr<GlobalContext> GlobalContext::Get() {
  if(!instance_) {
    LOG(ERROR)<<"The first call to Get should "
              <<"provide the sys/model conf path";
  }
  return instance_;
}
}  // namespace singa
