#ifndef INCLUDE_UTILS_CLUSTER_H_
#define INCLUDE_UTILS_CLUSTER_H_
#include <glog/logging.h>
#include <string>
#include <utility>
#include <memory>
#include <vector>
#include "proto/cluster.pb.h"

using std::shared_ptr;
using std::string;
using std::vector;

namespace singa {

/**
 * Cluster is a singlton object, which provides cluster configuations,
 * e.g., num workers/servers
 */
class Cluster {
 public:
  static shared_ptr<Cluster> Get();
  static shared_ptr<Cluster> Get(const ClusterProto& cluster,string hostfile,
    int procsid);

  void SetupGroups(const ClusterProto &cluster);
  void SetupFolders(const ClusterProto &cluster);

  const int nservers(){
    return cluster_.nservers();
  }
  const int nworkers() {
    return cluster_.nworkers();
  }
  bool AmIServer() {
    return procsid_>=nworkers()
      &&procsid_<nworkers()+nservers();
  }
  bool AmIWorker() {
    return procsid_>=0&&procsid_<nworkers();
  }
  int nprocs_per_group() {return cluster_.nprocs_per_group();}
  int nthreads_per_procs(){return cluster_.nthreads_per_procs();}
  int procsid() {return procsid_;}
  /**
   * Return the id of the worker within his group.
   */
  int groupid() {return procsid_/nprocs_per_group();}
  int ngroups() {return cluster_.nworkers()/nprocs_per_group();}
  int nthreads_per_group() {return nthreads_per_procs()*nprocs_per_group();}
  int groupid_of_procs(int procsid) {return procsid/nprocs_per_group();}
  //int procsid_of_thread(int threadid) {return threadid/nthreads_per_thread();}
    /**
   * thread id within a workring group, there are
   * procs_per_group()*nthreads_per_procs threads in one group.
   */
  int group_threadid(int local_threadid){
    return (procsid_%nprocs_per_group())*nthreads_per_procs()+local_threadid;
  }
  int group_procsid(int global_procsid){
    CHECK(global_procsid<cluster_.nworkers()&&global_procsid>=0);
    return global_procsid%nprocs_per_group();
  }
  int global_procsid(int local_threadid){
    return (procsid()/nprocs_per_group())*nprocs_per_group()+
      local_threadid/nthreads_per_procs();
  }

  int procsidOfGroupThread(int local_threadid){
    return local_threadid/nthreads_per_procs();
  }

  /**
   * procsid for the server that manages param for the worker of group_procsid
   */
  const string server_addr(int group_procsid){
    return addr(nworkers()+group_procsid%nprocs_per_group());
  }
  const string addr(int procsid) const{
    CHECK_GE(procsid,0);
    CHECK_LT(procsid,addr_.size());
    return addr_[procsid];
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
  Cluster(const ClusterProto &cluster, string hostfile, int procsid) ;

 private:
  int procsid_;
  // total number of processes started by mpi
  int nprocs_;
  int ngroups_;
  std::vector<std::string> addr_;
  // hostname
  std::string hostname_;
  // cluster config proto
  ClusterProto cluster_;
  // make this class a singlton
  static shared_ptr<Cluster> instance_;
};
}  // namespace singa

#endif  // INCLUDE_UTILS_GLOBAL_CONTEXT_H_
