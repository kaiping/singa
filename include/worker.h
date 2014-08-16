// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-04 11:14

#ifndef INCLUDE_WORKER_H_
#define INCLUDE_WORKER_H_
#include <map>
#include <memory>
#include "proto/model.pb.h"


#include "net/net.h"
#include "proto/model.pb.h"

namespace lapis {
/**
 * The Worker class which runs the training algorithm.
 * The first worker will initialize parameters of the Net, and put them into
 * the distributed memory/table.
 */

class Worker {
 public:
  Worker();
  void Run();

 private:
  void SetupNet(const int batchsize,
              const char flag,
              Net *net,
              const DataSourceProtos& protos,
              const std::map<std::string, int> &store_map);

  bool ShouldIDoValidation(int worker_id);
 private:
  std::shared_ptr<NetworkThread> mpi_;
  ModelController mc_;
};
}  // namespace lapis

#endif  // INCLUDE_WORKER_H_
