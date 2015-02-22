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

  const int nservers()const{
    return cluster_.nservers();
  }
  const int nworkers()const {
    return cluster_.nworkers();
  }
  bool AmIServer()const {
    return global_procsid_>=nworkers()
      &&global_procsid_<nworkers()+nservers();
  }
  bool AmIWorker()const {
    return global_procsid_>=0&&global_procsid_<nworkers();
  }
  int nprocs_per_group()const {return cluster_.nprocs_per_group();}
  int nthreads_per_procs()const{return cluster_.nthreads_per_procs();}
  int nthreads_per_server()const{return cluster_.nthreads_per_server();}
  int global_procsid()const {return global_procsid_;}
  /**
   * Return the id of the worker thread within his group.
   */
  int groupid() const{return global_procsid_/nprocs_per_group();}
  int ngroups() const{return cluster_.nworkers()/nprocs_per_group();}
  int nthreads_per_group() const{return nthreads_per_procs()*nprocs_per_group();}
  int group_threadid(int local_threadid)const{
    return group_procsid()*nthreads_per_procs()+local_threadid;
  }
  int group_procsid()const{
    return global_procsid_%nprocs_per_group();
  }
  int group_procsid(int group_threadid)const{
    return group_threadid/nthreads_per_procs();
  }

  const string server_addr() const{
    CHECK(AmIServer());
    return addr_.at(global_procsid());
  }
  /**
   * procsid for the server that manages param for the worker of group_procsid
   */
  const string server_addr(int server_procsid) const{
    CHECK_GE(server_procsid,0);
    CHECK_LT(server_procsid, nservers());
    return addr_.at(nworkers()+server_procsid);
  }
  const string group_thread_addr(int group_threadid) const{
    CHECK_GE(group_threadid,0);
    CHECK_LT(group_threadid,nthreads_per_group());
    return addr_.at(global_procsid_+group_procsid(group_threadid));
  }

  const string pub_port() const {
    return std::to_string(cluster_.start_port());
  }

  /**
   * pull port of ParameterManager
   */
  const int router_port() const {
    return cluster_.start_port()+1;
  }
  /**
   * pull port of Bridge layers.
   */
  const string pull_port(int k) const {
    return std::to_string(cluster_.start_port()+2+k);
  }
  bool synchronous()const {return cluster_.synchronous();}
  const string workerspace() {return cluster_.workspace();}
  const string visualization_folder(){
    return cluster_.workspace()+"/"+cluster_.vis_subfolder();
  }

  /**
   * bandwidth MB/s
   */
  float bandwidth() const {
    return cluster_.bandwidth();
  }
  const string hostname() const{return hostname_;}
 private:
  Cluster(const ClusterProto &cluster, string hostfile, int procsid) ;

 private:
  int global_procsid_;
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
