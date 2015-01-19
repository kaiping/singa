#ifndef INCLUDE_NET_SOLVER_H_
#define INCLUDE_NET_SOLVER_H_
#include <atomic>
#include <thread>
#include <string>
#include <vector>
#include <memory>

#include "proto/model.pb.h"
#include "model/net.h"
#include "utils/common.h"
#include "utils/shard.h"
#include "utils/global_context.h"
#include "core/table_delegate.h"
namespace singa {


/**
 * The base trainer class.
 * Child class has to implement the ::Init(), ::Train(), ::Validation(),
 * ::Test() and ::Checkpoint() functions.
 * Currently only one child trainer is implemented, i.e., SGDTrainer.
 * May support Newton-method based trainer later.
 */
class Solver {
  public:
    /**
     * init fields of the trainer
     * @param trainer_proto user configuration for the trainer
     */
    Solver(const SolverProto &proto);
    ~Solver();
    /**
     * Setup the solver, including setting the neural network and the
     * distributed table (content of the tuples, i.e., parameters).
     * @param np proto for the neural network.
     */
    Net* SetupNeuralNet(const NetProto& np);
    /**
     * Train the neraul network.
     * @param start_step start the training from this step.
     */
    void Train(Net* net, int start_step=0);
    /**
     * Test the perforance of the learned model.
     * @param phase kValidation or kTest.
     */
    Performance Test(Net* net, const Phase& phase);
    /**
     * Initialize neural network parameters and put them to the table.
     */
    void PopulateTableServer(Net* net);
    /**
     * Profiling the time cost of training one batch.
     * @param runs run this number of batches for average.
     */
    void TimeOneBatch(Net* net, int runs=10) ;
    /**
     * return the current training step
     * the ::Train() has been called such num of times
     */
    const int step() {
      return step_;
    }
 protected:
  void DebugInfo(Net* net);
  //void LocalUpdate(Param* param, int step);
  /**
   * train the model for one-minibatch by either backpropagation or contrastive divergence
   * @param net the Net object to be trained
   */
  Performance TrainOneBatch(Net* net, int step);
  /**
   * test performance on test dataset
   * @param net the current Net object
   */
  Performance TestOneBatch(Net *net, int step);
  /**
   * marshal the state of the trainer to google protobuf object, which will
   * later be dumped onto disk by ::Checkpoint()
   */
  void ToProto(SolverProto *proto);
  /**
   * return true if the stop condition is satisfied, e.g., the maximum number
   * of steps have been reached.
   */
  bool HasFinished(){
    if (step_ >= proto_.train_steps())
      return true;
    else
      return false;
  }

  const bool DisplayNow(const int step) {
    if (proto_.display_every_steps() > 0 && step >= proto_.display_after_steps()) {
      if ((step - proto_.display_after_steps()) % proto_.display_every_steps() == 0)
        return true;
    }
    return false;
  }
  const bool DisplayNow() {
    return DisplayNow(step_);
  }
  /**
    * @param step the ::Train() has been called step times.
    */
  const bool ValidateNow(const int step) {
    if (proto_.validation_every_steps() > 0 && step >= proto_.validation_after_steps()) {
      if ((step - proto_.validation_after_steps()) % proto_.validation_every_steps() == 0)
        return true;
    }
    return false;
  }
  const bool ValidateNow() {return ValidateNow(step_);}

  const bool TestNow(const int step) {
    if (proto_.test_every_steps() > 0 && step >= proto_.test_after_steps()) {
      if ((step - proto_.test_after_steps()) % proto_.test_every_steps() == 0)
        return true;
    }
    return false;
  }
  const bool TestNow() {return TestNow(step_);}

  void ReportPerformance(string prefix, Performance perf);
  /**
    * increase the step by one after each iteration
    * this operation is immediately called after the ::Train().
    */
  void IncStep() {
    step_++;
  }

 protected:
  //!< current phase, need this field to change the data sources for input layer
  //!< current training step, e.g., such num of mini-batches have been processed
  int step_;
  SolverProto proto_;
  Performance train_perf_, val_perf_, test_perf_;
  //!< path to the shard files
  string train_shard_, val_shard_, test_shard_;

  Phase phase_;
  TableDelegate* delegate_;
  std::shared_ptr<GlobalContext> context_;
};
/**
 * Prefech one batch of data/records from local shard (\class Shard) into the
 * input layers of the neural network.
 * It can be run in another thread to parallelize IO and computation.
 */
class Prefetcher {
 public:
  /**
   * @param path local shard file path.
   * @param _net pointer to the neural net which has been initialized.
   */
  Prefetcher(std::string path, Net* net, Phase phase);
  ~Prefetcher();
  /**
   * The main function, conducting prefeching task.
   * Override the operator for easy creating thread.
   */
  void operator()();

 private:
  /**
   * Read next record.
   * If reaches the end, then goto the begining of the shard and continue.
   */
  void NextRecord(Record* record);

 private:
  shard::Shard* shard_;
  Net* net_;
  Phase phase_;
};


}  // namespace lapis
#endif  // INCLUDE_NET_SOLVER_H_
