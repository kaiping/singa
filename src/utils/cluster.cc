#include <glog/logging.h>
#include "utils/cluster.h"
#include "utils/network.h"
#include "proto/cluster.pb.h"
#include "proto/worker.pb.h"

namespace singa {

std::shared_ptr<Cluster> Cluster::instance_;

void Cluster::CreateGroupComm(int start, int end,
    MPI_Comm* comm, MPI_Group* group){
  int *procs = new int[end-start];
  for (int i=start; i<end; i++)
    procs[i-start]=i;
  MPI_Group world_group;
  MPI_Comm_group(MPI_COMM_WORLD, &world_group);
  MPI_Group_incl(world_group, end-start, procs, group);
  MPI_Comm_create_group(MPI_COMM_WORLD, *group, 0, comm);
  delete procs;
}

Cluster::Cluster(const ClusterProto &cluster) {
	cluster_ = cluster;
  SetupGroups(cluster);
  SetupFolders(cluster);
}
void Cluster::SetupFolders(const ClusterProto &cluster){
  // create visulization folder
  mkdir(visualization_folder().c_str(),  S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
}
void Cluster::SetupGroups(const ClusterProto &cluster){
  char tmp[256];
  int len;
  MPI_Get_processor_name(tmp, &len);
  hostname_=string(tmp);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs_);
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
  CHECK(gid_!=-1||AmITableServer()) <<"gid "<<gid_<<" rank "<<rank_<<
    "start " << start << " end " << end << " istableserver "<<AmITableServer();
  // setup the group containing all workers
  if(gid_!=-1) {
    // setup worker's mpi group to be used in Barrier
    CreateGroupComm(groups_[gid_].front(), groups_[gid_].back()+1,
        &mycomm_, &mygroup_);
  }else{
    CreateGroupComm(cluster_.server_start(), cluster_.server_end(),
        &server_comm_, &server_group_);
    mycomm_=server_comm_;
    mygroup_=server_group_;
  }
  CreateGroupComm(start, end, &worker_comm_, &worker_group_);
  LOG(INFO)<<"Cluster Setup: "<<
    "Group id "<<gid_<<" rank "<<rank_<<" id within group "<<id_;
}
shared_ptr<Cluster> Cluster::Get(const ClusterProto& cluster){
  if(!instance_) {
    instance_.reset(new Cluster(cluster));
  }
  return instance_;
}
void Cluster::Finalize() {
  if(gid_!=-1){
    MPI_Comm_free(&mycomm_);
    MPI_Group_free(&mygroup_);
    MPI_Comm_free(&worker_comm_);
    MPI_Group_free(&worker_group_);
    LOG(ERROR)<<"Worker on "<<hostname_<< " is shutting down";
  }else{
    MPI_Comm_free(&server_comm_);
    MPI_Group_free(&server_group_);
    LOG(ERROR)<<"Server on "<<hostname_<< " is shutting down";
  }
}
shared_ptr<Cluster> Cluster::Get() {
  if(!instance_) {
    LOG(ERROR)<<"The first call to Get should "
              <<"provide the sys/model conf path";
  }
  return instance_;
}
}  // namespace singa
