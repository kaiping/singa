// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-14 13:32

#ifndef INCLUDE_NET_SOLVER_H_
#define INCLUDE_NET_SOLVER_H_
#include <string>
#include <vector>
#include "proto/model.pb.h"
#include "net/net.h"
#include "utils/common.h"


namespace lapis {
/**
 * Forward declaration of Net class
 */
class Net;
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
  /**
   * train the model for one-minibatch by either backpropagation or contrastive divergence
   * @param net the Net object to be trained
   */
  virtual Performance TrainOneBatch(Net* net);
  /**
   * test performance on validation dataset
   * @param net the Net object
   */
  virtual Performance ValidateOneBatch(Net *net);
  /**
   * test performance on test dataset
   * @param net the current Net object
  virtual Performance TestOneBatch(Net *net);
   */
  /**
   * marshal the state of the trainer to google protobuf object, which will
   * later be dumped onto disk by ::Checkpoint()
   */
  virtual void ToProto(SolverProto *proto);
  /**
   * return true if the stop condition is satisfied, e.g., the maximum number
   * of steps have been reached.
   */
  virtual bool HasFinished();
  /**
   * return true if it is time to do checkpoint
   * @param step the ::Train() has been called step times.
   */
  const bool CheckpointNow(const int step) {
    if (checkpoint_after_steps_ > 0 && step >= checkpoint_after_steps_) {
      if ((step - checkpoint_after_steps_) % checkpoint_every_steps_ == 0)
        return true;
    }
    return false;
  }
  const bool DisplayNow(const int step) {
    if (display_after_steps_ > 0 && step >= display_after_steps_) {
      if ((step - display_after_steps_) % display_every_steps_ == 0)
        return true;
    }
    return false;
  }
  const bool DisplayNow() {
    return DisplayNow(step_);
  }
  const bool CheckpointNow() {return CheckpointNow(step_);}
  /**
   * return true if it is time to do checkpoint
   * @param step the ::Train() has been called step times.
   */
  const bool ValidateNow(const int step) {
    if (validation_after_steps_ > 0 && step >= validation_after_steps_) {
      if ((step - validation_after_steps_) % validation_every_steps_ == 0)
        return true;
    }
    return false;
  }
  const bool ValidateNow() {return ValidateNow(step_);}

  int validation_steps() {
    return validation_steps_;
  }
  /**
   * increase the step by one after each iteration
   * this operation is immediately called after the ::Train().
   */
  void IncStep() {
    step_++;
  }
  /**
   * return the current training step
   * the ::Train() has been called such num of times
   */
  const int step() {
    return step_;
  }
  static int phase;
 protected:
  //! current phase, need this field to change the data sources for input layer
  //! current training step, e.g., such num of mini-batches have been processed
  int step_;

  //! start checkpoint after this num of steps
  int checkpoint_after_steps_;
  //! frequency for checkpoint
  int checkpoint_every_steps_;
  //! when recovered from the checkpoint, the step_ will be set to this value;
  int checkpoint_step_;
  //! path prefix (i.e., directory) for the checkpoint files
  std::string checkpoint_prefix_;

  //! start display after this num of steps
  int display_after_steps_;
  //! display frequency
  int display_every_steps_;
  //! path prefix (i.e., directory) for the displayed images
  std::string display_prefix_;

  //! start validation after this num of steps
  int validation_after_steps_;
  //! display frequency
  int validation_every_steps_;
  //! path prefix for Performance
  std::string perf_prefix_;
  int train_steps_;
  int validation_steps_;
};

}  // namespace lapis
#endif  // INCLUDE_NET_SOLVER_H_

