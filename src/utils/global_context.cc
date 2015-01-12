#include <glog/logging.h>
#include "utils/global_context.h"
#include "proto/cluster.pb.h"

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
  int start=cluster.worker_start();
  int end=cluster.worker_end();
  CHECK_LT(start, end);
  CHECK_LT(cluster.server_start(), cluster.server_end());

  CHECK_LE(cluster.server_end(), start);
  CHECK(cluster.group_size());
  for(int k=start, gid=0;k<end;gid++){
    vector<int> workers;
    for(int i=0;i<cluster.group_size();i++){
      if(k==rank_){
        gid_=gid;
        id_=i;
      }
      workers.push_back(k++);
    }
    groups_.push_back(workers);
  }
  CHECK(gid_!=-1||rank_==kCoordinator||AmITableServer())
    <<"gid "<<gid_<<" rank "<<rank_<< "start " << start << " end " << end << " istableserver "<<AmITableServer();
  // setup worker's mpi group to be used in Barrier
  if(gid_!=-1) {
    int gsize=cluster.group_size();
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

  // setup the group containing all workers
  int *members = new int[end-start];
  for (int i=start; i<end; i++)
	members[i-start]=i;
  MPI_Group world_group;
  MPI_Comm_group(MPI_COMM_WORLD,&world_group);
  MPI_Group_incl(world_group, end-start, members, &worker_group_);
  MPI_Comm_create_group(MPI_COMM_WORLD, worker_group_,0, &workergroup_comm_);

  LOG(INFO)<<"GlobalContext Setup: "<<
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
	MPI_Comm_free(&mpicomm_);
    MPI_Group_free(&mpigroup_);

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
