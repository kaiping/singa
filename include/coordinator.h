//  Copyright © 2014 Wei Wang, Anh. All Rights Reserved.
//  2014-08-07 12:53

#ifndef COORDINATOR_H_
#define COORDINATOR_H_
#include <unordered_set>
#include <vector>
#include <map>
#include "net/net.h"
#include "core/global-table.h"
#include "utils/global_context.h"
#include "utils/network_thread.h"
#include "proto/model.pb.h"
#include "model_controller/model.h"


namespace lapis {
// represent the (table, shard) tuple
struct TaskId {
  int table;
  int shard;

  TaskId(int t, int s): table(t), shard(s) {}
};

//  each memory server has a set of (table,shard) partitions
//  assigned to it. shardId is the same for all partitions
struct ServerState {
  int server_id;
  int shard_id;
  std::unordered_set<TaskId *> local_shards;

  ServerState(int id): server_id(id), shard_id(-1) {}
};

class Coordinator {
 public:
  Coordinator();
  void Run(bool load_data, bool do_run);
  /**
   * setup net, shape of darys are set, but not allocate mem
   */
  Net* SetupNetShape(const ModelProto& model);
  ~Coordinator();
 private:
  void InitDistributedStorage(bool load_data, const DataProto& proto);
  void InitTableServers(const std::map<int, GlobalTable*>& tables);
  void RunStandalone(const ModelProto& model);
  void RunOnCluster(const ModelProto& model);
  bool DoValidationOn(int worker_id);
  void Shutdown();

  const GroupConfig CreateGroups(int group_size);
  /**
   * insert parameters into table; the tuple is prepared with
   * parameter vector and sgd/adagrad meta info from solver
   * @threshold if sync mode, it is the group size; for async ,0
   */
  void FillParameterTable(int threshold,Solver* solver,  Net* net);
  void DistributePartition(const GroupConfig& conf, const vectro<ModelProto*> & protos);
  const vector<ModelProto*> PartitionModel(const ModelProto& model, Net* net);
  void InitTableDelegate(const SolverProto& solver);
 private:
  //  keep track of the table assignments, only to the memory servers
  std::vector<ServerState *> server_states_;
  shared_ptr<GlobalContext> context_;
  std::shared_ptr<NetworkThread> mpi_;
  TableDelegate* delegate_;
};
}  // namespace lapis
#endif  // COORDINATOR_H_
