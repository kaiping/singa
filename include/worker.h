// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-04 11:14

#ifndef INCLUDE_WORKER_H_
#define INCLUDE_WORKER_H_
#include <map>
#include <memory>

#include "core/table_server.h"
#include "core/table_delegate.h"
#include "net/net.h"
#include "net/solver.h"
#include "proto/model.pb.h"
#include "utils/common.h"

namespace lapis {
/**
 * The Worker class which runs the training algorithm.
 * The first worker will initialize parameters of the Net, and put them into
 * the distributed memory/table.
 */

class Worker {
 public:
  Worker(TableDelegate* delegate);
  ~Worker();
  void Run(bool do_run, bool time,  const SolverProto& solver_proto);

 private:
  bool ShouldIDoValidation(int step);
  bool AmIGroupLeader() {return leader_==id_;}
  void InitDistributedStorage();

  Performance Validate(Solver* solver, Net* net);
  void PrefetchData(int phase, Net *net);
  void Shutdown();
  void ReportPerformance(Performance perf);
  void InitGroupInfo();
 private:
  std::shared_ptr<NetworkThread> mpi_;
  TableServer *table_server_;
  TableDelegate* delegate_;
  vector<int> member_list_;
  int leader_;
  int id_;
};
}  // namespace lapis

#endif  // INCLUDE_WORKER_H_
