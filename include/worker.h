#ifndef INCLUDE_WORKER_H_
#define INCLUDE_WORKER_H_
#include <map>
#include <pthread.h>

#include "model/net.h"
#include "model/solver.h"
#include "proto/model.pb.h"

namespace singa {
/**
 * The Worker class which runs the training algorithm.
 * The first worker group will initialize parameters of the Net,
 * and put them into the distributed memory/table.
 *
 * May add more functions (e.g., communication with other workers/groups)
 */
class Worker {
 public:
  void Start(const ModelProto& model);
  void Resume();
};
}  // namespace singa

#endif  // INCLUDE_WORKER_H_
