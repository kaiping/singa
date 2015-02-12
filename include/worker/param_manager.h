#ifndef INCLUDE_WORKER_PARAM_MANAGER_H_
#define INCLUDE_WORKER_PARAM_MANAGER_H_

#include <czmq.h>
#include <thread>
#include "utils/param.h"
#include "utils/param_updater.h"
#include "worker/neuralnet.h"

#define kGradFrame 2
#define kDataFrame 4
#define kGradReady 8
#define kDataReady 16

namespace singa{
/**
 * ParamManager manages Param objects within the process.
 * It allocates the memory space for all Param objects that are used within this
 * process. It also synchronizes with parameter servers;
 * TODO syn with other processes for parameters shared across procs.
 */
class ParamManager{
 public:
  /**
   * Allocate memory for local Param objects of net and init network settings.
   */
  ParamManager(shared_ptr<NeuralNet> net, const UpdaterProto updater);

  /**
    * Initialize neural network parameters and put them to
    * distributed parameter table on parameter servers.
    * @param net, neural network
    */
  void PopulateServers(shared_ptr<Net> net);
  /**
   * randomlly init allocated parameters and set them ready
   */
  void InitParams();

    /**
   * Poll messages and conduct updates for parameters.
   */
  void Update(int step, shared_ptr<NeuralNet> net,  int threadID);
  /**
   * A loop which calls Update, running as a background thread.
   */
  void Run();
  void Stop(){
    running_=false;
  }
 protected:
  bool running_;
  //!< sgd updater
  shared_ptr<ParamUpdater> updater_;
  //!< a big param which allocates mem for all local params.
  Param param_;
  //!< map from param's owner ID to process whose ParamManager allocates
  //the owner param object.
  // map<int, int> paramOwnerID2procsID_;
  //!< map from param ID to Param poiner on local machine.
  //map<int, Param*> paramID2param_;
  map<int, vector<Param*>> ownerID2Params_;
  //!< aggregated updates for one param
  //map<int, int> aggregatedUpdates_;
  map<int, int> paramIDOffset_;
  // for leader PM to publish new parameters;
  // for worker PM to sub parameters from leader PM and;
  // for leader PM to pull grad from worker nodes;
  // for worker PM to push grad to leader PM;

  //!< publish param ready signal with param id
  //zsock_t pub_;
  //!< pull grad ready signal with param id (or addr?)
  //zsock_t pull_;
  std::mutex mtx_;
  std::condition_variable cv_;
  int step_;

  //!< sub updates/grad from PS
  zsock_t sub_;
  //!< push updates/grad to PS
  zsock_t push_;
  zpoller_t *poller_;

  int timeout_;
  int updateLimit_;
  int syncfreq_;
};
}
#endif // INCLUDE_WORKER_PARAM_MANAGER_H_

