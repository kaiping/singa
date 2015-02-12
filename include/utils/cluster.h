#ifndef INCLUDE_UTILS_CLUSTER_H_
#define INCLUDE_UTILS_CLUSTER_H_
#include <glog/logging.h>
#include <string>
#include <utility>
#include <memory>
#include <vector>
#include <mpi.h>
#include "proto/cluster.pb.h"

using std::shared_ptr;
using std::string;
using std::vector;

namespace singa {

/**
 * Cluster is a singlton object, which provides cluster configuations,
 * e.g., num workers/servers and MPI groups for coordination, e.g, Barrier
 */
class Cluster {
 public:
  static shared_ptr<Cluster> Get();
  static shared_ptr<Cluster> Get(const ClusterProto& cluster);
  // free my mpi group and mpi communicator

  void SetupGroups(const ClusterProto &cluster);
  void SetupFolders(const ClusterProto &cluster);

  const int num_servers_procs(){
    return cluster_.nservers();
  }
  const int num_worker_procs() {
    return cluster_.nworkers();
  }
  bool AmIServer() {
    return procsID_>=num_worker_procs()
      &&procsID_<num_worker_procs()+num_servers_procs();
  }
  bool AmIWorker() {
    return procsID_>=0&&procsID_<num_worker_procs();
  }
  /**
   * Return the id of the worker within his group.
   */
  int groupID() {return procsID/nprocs_per_group();}
  int ngroups() {return cluster_.worker_size()/nprocs_per_group_;}
  int procsID() {return procsID_;}
  int nprocs_per_group() {return cluster_.nprocs_per_group();}
  int nthreads_per_procs(){return cluster_.nthreads_per_procs();}
  int nthreads_per_group() {return nthreads_per_procs()*nprocs_per_group();}
  int groupID_of_procs(int procsID) {return procsID/nprocs_per_group();}
  int procsID_of_thread(int threadID) {return threadID/nthreads_per_thread();}
  int groupID_of_thread(int threadID) {
    return groupID_of_procs(procsID_of_thread(threadID));
  }
  /**
   * thread ID within a workring group, there are
   * procs_per_group()*nthreads_per_procs threads in one group.
   */
  int group_threadID(int local_threadID){
    (procsID_%nprocs_per_group())*nthreads_per_procs()+local_threadID;
  }
  /**
   * thread ID among all worker nodes/procs
   */
  int global_threadID(int local_threadID){
    return procsID()*nthreads_per_procs()+local_threadID;
  }
  int group_procsID(int global_procsID){
    CHECK(global_procsID<cluster_.nworkers()&&global_procsID>=0);
    return global_procsID%nprocs_per_group();
  }
  int global_procsID(int local_threadID){
    return (procsID/nprocs_per_group)*nprocs_per_group+
      local_threadID/nthreads_per_procs();
  }

  /**
   * procsID for the server that manages param for the worker of group_procsID
   */
  const string server_addr(int group_procsID){
    return addr(num_worker_procs()+group_procsID%nprocs_per_group());
  }
  const string addr(int procsID) const{
    CHECK(procsID>=0&&procsID<addr_.size());
    return addr_[procsID];
  }
  const string pub_port() const {
    return std::to_string(cluster_.start_port());
  }
  /**
   * pull port of ParameterManager
   */
  const string pull_port() const {
    return std::to_string(cluster_.start_port()+1);
  }
  /**
   * pull port of Bridge layers.
   */
  const string pull_port(int k) const {
    return std::to_string(cluster_.start_port()+2+k);
  }
  bool synchronous() {return cluster_.synchronous();}
  const string workerspace() {return cluster_.workspace();}
  const string visualization_folder(){
    return cluster_.workspace()+"/"+cluster_.vis_subfolder();
  }
  const string hostname(){return hostname_;}
 private:
  Cluster(const ClusterProto &cluster, string hostfile, int procsID) ;

 private:
  int procsID_;
  // total number of processes started by mpi
  int nprocs_;
  int ngroups_;
  // hostname
  std::string hostname_;
  // cluster config proto
  ClusterProto cluster_;
  // make this class a singlton
  static shared_ptr<Cluster> instance_;
};
}  // namespace singa

#endif  // INCLUDE_UTILS_GLOBAL_CONTEXT_H_
