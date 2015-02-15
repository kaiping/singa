#ifndef INCLUDE_WORKER_PARAM_MANAGER_H_
#define INCLUDE_WORKER_PARAM_MANAGER_H_

#include <czmq.h>
#include <thread>
#include <mutex>
#include <condition_variable>
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
  ParamManager(shared_ptr<NeuralNet> net, const UpdaterProto& updater);

  /**
   * called by local worker threads;
   * can be implemented in hogwild way, i.e., done asynchornously; or in batch
   * mode, i.e., wait until all threads update for this param is ready.
   * can be done by the stub thread or the calling thread
   */
  void UpdateParam(Param* param, int step, int threadid);
  /**
   * call UpdateParam to update all params used by the calling thread
   * blocked until all params are updated.
  void UpdateParams(int step, int threadid);
   */
  /**
   * will be blocked if the param is not updated.
   */
  void WaitUpdate(Param* param, int step, int threadid);
  /**
    * Initialize neural network parameters and put them to
    * distributed parameter table on parameter servers.
    * @param net, neural network
    */
  void SendParamsToServers(){}
  void GetParamsFromServers(){} // will be blocked until recv all parameters.
  /**
   * randomlly init allocated parameters and set them ready */
  void InitParams();

    /**
   * Poll messages and conduct updates for parameters.
   */
  void Update(int step, int threadid);
  /**
   * A loop which calls Update, running as a background thread.
   */
  void Run(int step);
  void Stop(){
    running_=false;
  }
  void SyncWithPS(int step);

  void HandleParamUpdate(zmsg_t* msg){}
  void HandleLocalMsg(int paramid, zmsg_t* msg){}
  void HandlePSMsg(int paramid){}

 protected:
  bool hogwild_;
  bool running_;
  int sync_frequency_;
  shared_ptr<NeuralNet> net_;
  //!< sgd updater
  shared_ptr<ParamUpdater> updater_;
  //!< a big param which allocates mem for all local params.
  Param param_;
  //!< map from param's owner id to process whose ParamManager allocates
  //the owner param object.
  // map<int, int> paramOwnerid2procsid_;
  //!< map from param id to Param poiner on local machine.
  //map<int, Param*> paramid2param_;
  map<int, vector<Param*>> ownerid2Params_;
  //!< aggregated updates for one param
  map<int, int> aggregatedUpdates_;
  map<int, int> paramid2Offset_;
  map<Param*, int> param2version_;
  std::mutex mtx_;
  //std::condition_variable cv_;


};
}
#endif // INCLUDE_WORKER_PARAM_MANAGER_H_

