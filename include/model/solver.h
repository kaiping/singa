#ifndef INCLUDE_MODEL_SOLVER_H_
#define INCLUDE_MODEL_SOLVER_H_
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
 * Solver class control the training, validation and test.
 * The first group conducts the validation and test.
 * Solver owns a TableDelegate to communicate with parameter servers.
 * Currently only support Back-Propagation. TODO add Contrastive Divergence.
 */
class Solver {
 public:
  /**
    * init fields of the trainer
    * @param proto user configuration for the solver
    */
  Solver(const SolverProto &proto);
  ~Solver();
  /**
    * marshal the state of the trainer to google protobuf object, which will
    * later be dumped onto disk by ::Checkpoint()
    */
  void ToProto(SolverProto *proto);
  /**
    * Train the neraul network.
    * @param net, neural network
    * @param start_step start the training from this step.
    */
  void Train(Net* net, int start_step=0);
  /**
    * Test the perforance of the learned model on validation or test dataset.
    * Test is done by the first group.
    * @param net, neural network
    * @param phase kValidation or kTest.
    */
  Performance Test(Net* net, const Phase& phase);
  /**
    * Setup the neural network and the
    * @param np proto for the neural network.
    */
  Net* SetupNeuralNet(const NetProto& np);
  /**
    * Initialize neural network parameters and put them to
    * distributed parameter table on parameter servers.
    * @param net, neural network
    */
  void PopulateTableServer(Net* net);
  /**
    * Profiling the time cost of training one batch.
    * @param net, neural network
    * @param runs run this number of batches for average.
    */
  void TimeOneBatch(Net* net, int runs=10) ;
  /**
    * return the current training step on this node
    * the ::Train() has been called such num of times
    */
  const int step() {
    return step_;
  }

 protected:
  /**
   * Print Norm1 of data and grad of each Layer and parameter.
   * @param net, neural network
   */
  void DebugInfo(Net* net);
  /**
   * Train for mini-batch
   * @param net the current Net object
   * @return loss
   */
  Performance TrainOneBatch(Net* net, int step);
  /**
   * Test for one mini-batch
   * @param net the current Net object
   * @return loss
   */
  Performance TestOneBatch(Net *net, int step);
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

  /**
   * Check is it time to display training info, e.g., loss and precison.
   */
  const bool DisplayNow(const int step) {
    if (proto_.display_frequency() > 0
        && step >= proto_.display_after_steps()) {
      if ((step - proto_.display_after_steps())
          % proto_.display_frequency() == 0)
        return true;
    }
    return false;
  }
  /**
   * Check is it time to display training info, e.g., loss and precison.
   */
  const bool DisplayNow() { return DisplayNow(step_); }
  /**
   * Check is it time to do validation.
   * @param step the ::Train() has been called step times.
   */
  const bool ValidateNow(const int step) {
    if (proto_.validation_frequency() > 0
        && step >= proto_.validation_after_steps()) {
      if ((step - proto_.validation_after_steps())
          % proto_.validation_frequency() == 0)
        return true;
    }
    return false;
  }
  const bool ValidateNow() {return ValidateNow(step_);}
  /**
   * Check is it time to do validation.
   * @param step the ::Train() has been called step times.
   */
  const bool TestNow(const int step) {
    if (proto_.test_frequency() > 0 && step >= proto_.test_after_steps()) {
      if ((step - proto_.test_after_steps()) % proto_.test_frequency() == 0)
        return true;
    }
    return false;
  }
  const bool TestNow() {return TestNow(step_);}

  /**
   * Print aggregated performance for training/test/validation
   * @param prefix, 'Test' or 'Train' or 'Val' (for validation)
   */
  void ReportPerformance(string prefix, Performance perf);
  /**
    * increase the step by one after each iteration
    * this operation is immediately called after the ::Train().
    */
  void IncStep() {
    step_++;
  }

 protected:
  //!< current training step, this num of mini-batches have been processed
  int step_;
  //!< path to the shard files
  string train_shard_, validation_shard_, test_shard_;
  //!< user configuration (updated accroding to group id).
  SolverProto proto_;

  //!< current phase, either kTrain, kTest or kValidation
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
   * @param random_skip skip some records used in training phase
   */
  Prefetcher(std::string path, Net* net, Phase phase, bool random_skip=false);
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
