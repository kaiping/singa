#ifndef INCLUDE_WORKER_H_
#define INCLUDE_WORKER_H_
#include <map>
#include <pthread.h>

#include "worker/neuralnet.h"
#include "worker/param_manager.h"
#include "proto/model.pb.h"
#include "utils/cluster.h"

namespace singa {
class Performance{
 public:
  explicit Performance(shared_ptr<NeuralNet> net);
  /**
   * aggregate metrics from LossLayer
   */
  void Update();
  void Reset();
  string ToString();
 private:
  vector<vector<float>> metric_;
  vector<string> name_;
  shared_ptr<NeuralNet> net_;
  int counter_;
};

/**
 * Executor runs as a thread which owns one partition of the neuralnet
 */
class Executor{
 public:
  Executor():local_threadid_(0){}
  virtual ~Executor();
  /**
    * threadid is the thread id within a working group
    */
  Executor(int local_threadid,
      const ModelProto& model,
      shared_ptr<Cluster> cluster,
      shared_ptr<NeuralNet> train_net,
      shared_ptr<NeuralNet> test_net=nullptr,
      shared_ptr<NeuralNet> validation_net=nullptr);
  void Setup(int local_threadid, const ModelProto& model);
  virtual void Run(ParamManager*pm, int start_step=0);
  /**
    * Fetchdata by calling DataLayer and ParserLayer of the net.
    * This function is usually called by launcing a new thread as prefetching.
    */
  static void PrefetchData(const vector<DataLayer*>& datalayers, bool training,
      int steps=1);

  /**
    * check validation/test firstly, then TrainOneBatch
    * Performance collects performance for the whole neuralnet.
    * Hence, no need to collect performance in every thread.
    * Only the main thread will pass none null perf.
    */
  void RunOneBatch(int step, Performance* perf=nullptr);

  /**
    * Train one mini-batch.
    * Test/Validation and Display is done before training.
    */
  void TrainOneBatch(int step);

  /**
    * Test the perforance of the learned model on validation or test dataset.
    * Test is done by the first group.
    * @param net, neural network
    * @param phase kValidation or kTest.
    */
  void Test(shared_ptr<NeuralNet> net, int nsteps, bool dispperf);
  void Pull(zsock_t* pull, shared_ptr<NeuralNet> net);

  void Forward(shared_ptr<NeuralNet> net, bool training);
  void Backward(shared_ptr<NeuralNet> net);
  /**
    * Profiling the time cost of training one batch.
    */
  string TimerInfo(){
    char buf[1024];
    float ticks=ticks_*1000;
    float tf=tForward_/ticks, tb=tBackward_/ticks,
          td=tSyncData_/ticks, tp=tSyncParam_/ticks;
    float total=tf+tb+td+tp;
    sprintf(buf,
        "Total\t%6.2f\tforward\t%6.2f\tbackward\t%6.2f\t\
        syncdata\t%6.2f\tsyncparam\t%6.2f\n", total,tf,tb, td,tp);
    tForward_=0;
    tBackward_=0;
    tSyncData_=0;
    tSyncData_=0;
    ticks_=0;
    return string(buf);
  }
  /**
    * Check is it time to display training info, e.g., loss and precison.
    */
  const bool DisplayNow(const int step) const {
    return (cluster_->groupid()==0
        &&modelproto_.display_frequency() > 0
        && step >= modelproto_.display_after_steps()
        && ((step - modelproto_.display_after_steps())
          % modelproto_.display_frequency() == 0));
  }

  /**
    * return true if the stop condition is satisfied, e.g., the maximum number
    * of steps have been reached.
    */
  const bool StopNow(const int step) const{
    return (step >= modelproto_.train_steps());
  }

  /**
    * Check is it time to do test.
    * @param step the ::Train() has been called this num times.
    */
  const bool TestNow(const int step) const{
    return (cluster_->groupid()==0
        && modelproto_.test_frequency() > 0
        && step >= modelproto_.test_after_steps()
        && ((step - modelproto_.test_after_steps())
          % modelproto_.test_frequency() == 0));
  }
  /**
    * Check is it time to do validation.
    * @param step the ::Train() has been called step times.
    */
  const bool ValidateNow(const int step) {
    return (cluster_->groupid()==0
        && modelproto_.validation_frequency() > 0
        && step >= modelproto_.validation_after_steps()
        && ((step - modelproto_.validation_after_steps())
          % modelproto_.validation_frequency() == 0));
  }

 protected:
  int local_threadid_;
  ModelProto modelproto_;
  shared_ptr<Cluster> cluster_;
  //!< thread for prefetching training data.
  std::thread prefetch_thread_;
  int step_;

  float tForward_, tBackward_, tSyncData_, tSyncParam_;
  int ticks_;

  zsock_t* pull_;
  map<int, zsock_t*> push_;
  shared_ptr<NeuralNet> train_net_, test_net_, validation_net_;
  vector<DataLayer*> localDataLayers_;
};

/**
 * The Worker class which runs the training algorithm.
 * The first worker group will initialize parameters of the Net,
 * and put them into the distributed memory/table.
 * It owns a Delegate to communicate with parameter servers.
 *
 * TODO May add more functions (e.g., communication with other workers/groups).
 */
class Worker : public Executor{
 public:
  explicit Worker(shared_ptr<Cluster> cluster);
  /**
   * start training from scratch.
   * setup training/test/validation neuralnets, then call Run().
   */
  void Start(ModelProto model);
  /**
   * TODO Resume from snapshot
   */
  void Resume();

 protected:
  /**
    * Main function of Worker.
    * 1. Train the neuralnet step by step, test/validation is done periodically.
    * 2. TODO Communicate with others, e.g., zookeeper, after every step.
    * @param net, neural network
    * @param start_step start the training from this step.
    */
  virtual void Run(ParamManager*pm, int start_step=0);
  /**
   * Setup the neural network for training, test or validation.
   * Weights for test/validation net can share those from training after
   * setup (done outside of this funcion).
   * @param np proto for the neural network.
   */
  shared_ptr<NeuralNet> SetupNeuralNet(const NetProto& np, bool prefetch,
      Phase phase);
};
}  // namespace singa

#endif  // INCLUDE_WORKER_H_
