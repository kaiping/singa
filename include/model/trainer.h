// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-14 13:32

#ifndef INCLUDE_MODEL_TRAINER_H_
#define INCLUDE_MODEL_TRAINER_H_
#include <string>
#include <vector>
#include "proto/lapis.pb.h"
#include "model/net.h"
#include "model/data_source.h"
#include "worker/model_controller.h"

namespace lapis {
/**
  * there are three phases, i.e., training, validation, and test
  */
enum class TrainPhase {
  kTrain = 0,
  kValidation = 1,
  kTest = 2
};
enum class TrainAlgorithm {
  kBackPropagation,
  kCD,
  kPCD
};

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
  virtual void Init(const TrainerProto &trainer_proto);
  /**
   * train the model by either backpropagation or contrastive divergence
   * @param net the Net object to be trained
   * @param step the current training step, e.g., id of the mini-batch
   */
  virtual void Train(Net *net, const int step) = 0;
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
   * marshal the state of the trainer to google protobuf object, which will
   * later be dumped onto disk by ::Checkpoint()
   */
  virtual void ToProto(TrainerProto *proto);
  /**
   * return true if the stop condition is satisfied, e.g., the maximum number
   * of steps have been reached.
   * @param step such number of iterations have been processed
   */
  virtual bool HasFinished(const int step);
  /**
   * return true if it is time to do checkpoint
   * @param step the ::Train() has been called step times.
   */
  inline const bool CheckpointNow(const int step);
  /**
   * increase the step by one after each iteration
   * this operation is immediately called after the ::Train().
   */
  inline void IncStep();
  /**
   * return the current training step
   * the ::Train() has been called such num of times
   */
  inline const int Step();
 protected:
  //! current phase, need this field to change the data sources for input layer
  TrainPhase phase_;
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

  //! data providers for training, see DataSource
  std::vector<DataSource *> train_data_;
  //! data providers for validation, see DataSource
  std::vector<DataSource *> validation_data_;
  //! data providers for test, see DataSource
  std::vector<DataSource *> test_data_;

  //! path prefix for Performance
  std::string perf_prefix_;

  //! ModelController to provide parameters and input features
  ModelController model_controller_;
};

}  // namespace lapis
#endif  // INCLUDE_MODEL_TRAINER_H_

