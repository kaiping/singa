#include <glog/logging.h>
#include <memory>
#include "worker.h"
#include "proto/model.pb.h"
#include "utils/cluster.h"

namespace singa {
void Worker::Resume() {
  // TODO implement resume from snapshot
}

void Worker::Start(const ModelProto& model){
  LOG(ERROR)<<"Worker on "<<Cluster::Get()->hostname()<<" is starting...";
  Solver solver(model.solver());
  Net* net=solver.SetupNeuralNet(model.net());
  if(Cluster::Get()->group_id()==0)
    solver.PopulateTableServer(net);
  // todo, two ways to syn all workers
  // 1. create MPI communicator for all workers, and call MPI_Barrier for
  // this communicator
  // 2. handle_get returns false if the key of get() is not found in the
  // table, i.e., parameters have not been inserted
  MPI_Barrier(Cluster::Get()->worker_comm());
  //solver.Train(net);
  solver.TimeOneBatch(net, 5);
  //LOG(ERROR)<<"Worker on "<<hostname_<< " is shutting down";
}
}  // namespace singa
