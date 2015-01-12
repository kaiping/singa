#include <glog/logging.h>
#include <memory>
#include "worker.h"
#include "proto/model.pb.h"
#include "utils/global_context.h"
#include "net/solver.h"
#include "net/net.h"

namespace singa {

void Worker::Resume() {
  // TODO implement resume from snapshot
}

void Worker::Start(const Model& model){
    Solver solver(model.solver());
    Net* net=solver.SetupNeuralNet(model.net());
    if(GlobalContext::Get()->group_id()==0)
      solver.PopulateTableServer(net);
    // todo, two ways to syn all workers
    // 1. create MPI communicator for all workers, and call MPI_Barrier for
    // this communicator
    // 2. handle_get returns false if the key of get() is not found in the
    // table, i.e., parameters have not been inserted
    LOG(ERROR)<<"Worker starting...";
    solver.Train();
  LOG(ERROR)<<"Worker shutting down";
}
}  // namespace singa
