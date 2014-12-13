// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-06-28 14:58
#ifndef INCLUDE_UTILS_GLOBAL_CONTEXT_H_
#define INCLUDE_UTILS_GLOBAL_CONTEXT_H_
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

namespace lapis {

/**
 * Global Context is a singlton object, which provides cluster configuations,
 * e.g., num workers/servers.
 * It also provides the MPI group of a worker group for coordination
 * e.g, Barrier
 */
class GlobalContext {
 public:
  // assume the rank of coordinator is num_procs-1
  static int kCoordinator;
  static shared_ptr<GlobalContext> Get();
  static shared_ptr<GlobalContext> Get(const Cluster& cluster);
  // free my mpi group and mpi communicator
  void Finalize();

  const int num_table_servers() {
    return cluster_.server_end()-cluster_.server_start();
  }
  const int server_start() {
    return cluster_.server_start();
  }
  const int server_end() {
    return cluster_.server_end();
  }
  const int num_servers(){
    return cluster_.server_end()-cluster_.server_start();
  }
  const int num_procs() {
    return num_procs_;
  }
  const int num_workers() {
    return cluster_.worker_end()-cluster_.worker_start();
  }
  const bool IsTableServer(int rank) {
    return rank>=server_start()&&rank<server_end();
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
  const string data_folder() {return cluster_.data_folder();}
  const MPI_Comm& mpicomm() {return mpicomm_;}
  vector<int> MembersOfGroup(int gid) {return groups_[gid];}
  const vector<vector<int>>& groups() {return groups_;}

 private:
  GlobalContext(const Cluster& cluster);

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
  Cluster cluster_;
  // my mpi group, for MPI Barrier
  MPI_Group mpigroup_;
  // my mpi communicator, for MPI Barrier
  MPI_Comm mpicomm_;
  // make this class a singlton
  static shared_ptr<GlobalContext> instance_;
};
}  // namespace lapis

#endif  // INCLUDE_UTILS_GLOBAL_CONTEXT_H_
