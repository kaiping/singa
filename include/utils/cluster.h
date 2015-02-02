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
  // assume the rank of coordinator is num_procs-1
  static int kCoordinator;
  static shared_ptr<Cluster> Get();
  static shared_ptr<Cluster> Get(const ClusterProto& cluster);
  // free my mpi group and mpi communicator
  void Finalize();

  void SetupGroups(const ClusterProto &cluster);
  void SetupFolders(const ClusterProto &cluster);

  const int num_table_servers() {
    return num_servers();
  }
  const int server_start() {
    return cluster_.server_start();
  }
  const int server_end() {
    return cluster_.server_end();
  }
  const int worker_start() {
    return cluster_.worker_start();
  }
  const int worker_end() {
    return cluster_.worker_end();
  }

  const int num_servers(){
    if(cluster_.has_server_start()&&cluster_.has_server_end())
      return cluster_.server_end()-cluster_.server_start();
    else return 0;
  }
  const int num_procs() {
    return num_procs_;
  }
  const int num_workers() {
    return cluster_.worker_end()-cluster_.worker_start();
  }
  const bool IsTableServer(int rank) {
    if(cluster_.has_server_start()&&cluster_.has_server_end())
      return rank>=server_start()&&rank<server_end();
    else return false;
  }
  const bool AmICoordinator() {
    return rank_==kCoordinator;
  }
  bool AmITableServer() {
    return IsTableServer(rank_);
  }
  bool AmIWorker() {
    return rank_>=cluster_.worker_start()&&rank_<cluster_.worker_end();
  }
  /**
   * Return the id of the worker within his group.
   */
  int worker_id() {return id_;}
  int group_id() {return gid_;}
  int num_groups() {return groups_.size();}
  int group_size() {return cluster_.group_size();}
  bool synchronous() {return cluster_.synchronous();}
  int rank() {return rank_;}
  //bool checkpoint_enabled() {return cluster_.checkpoint_enabled();}
  //int checkpoint_freq() {return cluster_.checkpoint_freq();}
  //int checkpoint_after() {return cluster_.checkpoint_after();}
  const string workerspace() {return cluster_.workspace();}
  const string visualization_folder(){
    return cluster_.workspace()+"/"+cluster_.vis_subfolder();
  }
  vector<int> MembersOfGroup(int gid) {return groups_[gid];}
  const vector<vector<int>>& groups() {return groups_;}
  /**
   * return the MPI_Comm of the all workers
   */
  const MPI_Comm& worker_comm() {return worker_comm_;}
  /**
   * return the MPI_Comm for my group
   */
  const MPI_Comm& mycomm(){return mycomm_;}
  /**
   * return the MPI_Comm for all servers
   */
  const MPI_Comm& server_comm(){return server_comm_;}

  const string hostname(){return hostname_;}
 private:
  Cluster(const ClusterProto& cluster);
  void CreateGroupComm(int start, int end, MPI_Comm* comm, MPI_Group* group);

 private:
  // mpi rank, global ID
  int rank_;
  // ID of worker within group, starts from 0; -1 for servers
  int id_;
  // worker group id, start from 0; -1 for servers
  int gid_;
  // total number of processes started by mpi
  int num_procs_;
  // hostname
  std::string hostname_;
  // ranks of workers for each group
  vector<vector<int>> groups_;
  // cluster config proto
  ClusterProto cluster_;
  // my mpi group, for MPI Barrier
  MPI_Group mygroup_;
  // my mpi communicator, for MPI Barrier
  MPI_Comm mycomm_;
  // group for all workers
  MPI_Group worker_group_;
  // mpi comm for all workers
  MPI_Comm worker_comm_;
  // mpi group for all servers
  MPI_Group server_group_;
  // mpi comm for all servers
  MPI_Comm server_comm_;

  // make this class a singlton
  static shared_ptr<Cluster> instance_;
};
}  // namespace singa

#endif  // INCLUDE_UTILS_GLOBAL_CONTEXT_H_
