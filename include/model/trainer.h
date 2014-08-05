// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-14 13:32

#ifndef INCLUDE_MODEL_TRAINER_H_
#define INCLUDE_MODEL_TRAINER_H_
#include <string>
#include <vector>
#include "proto/model.pb.h"
#include "model/net.h"
#include "disk/data_source.h"
#include "model_controller/model.h"
#include "utils/common.h"

namespace lapis {
/**
  * there are three phases, i.e., training, validation, and test
  */
enum class Phase {
  kInit = 0,
  kTrain = 1,
  kValidation = 2,
  kTest = 3
};
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
class Trainer {
 public:
  /**
   * init fields of the trainer
   * @param trainer_proto user configuration for the trainer
   */
  virtual void Init(const TrainerProto &proto, ModelController *mc);
  virtual ~Trainer();
  /**
   * Init a DataSource object based on DataSourceProto
   */
  static void InitDataSource(
    const ::google::protobuf::RepeatedPtrField<DataSourceProto> &protos,
    std::vector<DataSource *> *sources);

  /**
   * train the model by either backpropagation or contrastive divergence
   * @param net the Net object to be trained
   * @param step the current training step, e.g., id of the mini-batch
   */
  virtual void Train(const int step,Net* net,
                     const char flag=kAllocData|kAllocParam|kInitParam)=0;
  /**
   * test performance on validation dataset
   * @param net the Net object
   */
  virtual void Validate(Net *net) = 0;
  /**
   * test performance on test dataset
   * @param net the current Net object
   */
  virtual void Test(Net *net) = 0;
  /**
   * Run the trainer
   * @param net the neural network
   */
  virtual void Run(const char flag, Net *net);

  /**
   * marshal the state of the trainer to google protobuf object, which will
   * later be dumped onto disk by ::Checkpoint()
   */
  virtual void ToProto(TrainerProto *proto);
  /**
   * return true if the stop condition is satisfied, e.g., the maximum number
   * of steps have been reached.
   * @param step such number of iterations have been processed
   */
  virtual bool HasFinished(const int step) = 0;
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
  /**
   * return true if it is time to do checkpoint
   * @param step the ::Train() has been called step times.
   */
  const bool ValidateNow(const int step) {
    if (validate_after_steps_ > 0 && step >= validate_after_steps_) {
      if ((step - validate_after_steps_) % validate_every_steps_ == 0)
        return true;
    }
    return false;
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

  static Phase phase;
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
  int validate_after_steps_;
  //! display frequency
  int validate_every_steps_;

  //! data providers for training, see DataSource
  std::vector<DataSource *> train_data_;
  //! data providers for validation, see DataSource
  std::vector<DataSource *> validation_data_;
  //! data providers for test, see DataSource
  std::vector<DataSource *> test_data_;

  //! path prefix for Performance
  std::string perf_prefix_;

  //! Call Train and Validate() if true
  bool do_train_;
  //! Call Test if true
  bool do_test_;

  //! ModelController to provide parameters and input features
  ModelController *model_controller_;
};

}  // namespace lapis
#endif  // INCLUDE_MODEL_TRAINER_H_

