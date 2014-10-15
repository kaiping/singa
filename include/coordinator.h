//  Copyright © 2014 Wei Wang, Anh. All Rights Reserved.
//  2014-08-07 12:53

#ifndef COORDINATOR_H_
#define COORDINATOR_H_
#include <unordered_set>
#include <vector>
#include <map>
#include <memory>

#include "net/net.h"
#include "utils/global_context.h"
#include "utils/network_thread.h"
#include "utils/common.h"
#include "proto/model.pb.h"
#include "net/solver.h"

using std::vector;
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

typedef struct WorkerState_{
  int worker_id; // rank of the  worker
  int step; // current step of the worker
  bool alive; // state: true if alive; false if dead
} WorkerState;

class Coordinator {
 public:
  explicit Coordinator(const shared_ptr<GlobalContext>& gc);
  void Start(const ModelProto& model);
  void Resume();
  void Shutdown();
  ~Coordinator();
 private:
  void Run();
  void InitTableServers(const std::map<int, GlobalTable*>& tables);
  /**
    * setup net, shape of darys are set, but not allocate mem
    */
  Net* SetupNetShape(const ModelProto& model);

  /**
    * insert parameters into table; the tuple is prepared with
    * parameter vector and sgd/adagrad meta info from solver
    * @threshold if sync mode, it is the group size; for async ,0
    */
  void DistributePartition(const vector<NetProto> & protos);
  const vector<NetProto> PartitionNet(Net* net);
 private:
  //  keep track of the table assignments, only to the memory servers
  std::vector<ServerState *> server_states_;
  shared_ptr<GlobalContext> context_;
  std::shared_ptr<NetworkThread> mpi_;
};
}  // namespace lapis
#endif  // COORDINATOR_H_
