// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-11 13:53

#ifndef INCLUDE_MODEL_SGD_TRAINER_H_
#define INCLUDE_MODEL_SGD_TRAINER_H_

#include <vector>
#include "model/param.h"
#include "model/trainer.h"
#include "proto/model.pb.h"

namespace lapis {
/**
 * Train the Net by stochastic gradient descent method,
 * it inherits from the base Trainer
 */
class SGDTrainer : public Trainer {
 public:
  virtual void Init(const TrainerProto &trainer_proto);
  virtual void Train(Net *net, const int step);
  virtual void Validate(Net *net);
  virtual void Test(Net *net);
  virtual void ToProto(TrainerProto *proto);
  virtual bool HasFinished(const int step);

  float learning_rate() const {return learning_rate_;}
  float momentum() const {return momentum_;}
  float weight_decay() const  {return weight_decay_;}
  void UpdateHyperParams(const int step);
  ~SGDTrainer();
 protected:
  float UpdateHyperParam(int step, SGDProto_ChangeProto change,
                         int change_steps, float base_val, float final_val);

  void BackPropagation(Net *net, const int step);

 private:
  float learning_rate_, momentum_, weight_decay_;
  SGDProto sgd_proto_;
};

}  // namespace lapis
#endif  // INCLUDE_MODEL_SGD_TRAINER_H_
