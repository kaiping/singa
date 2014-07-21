// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-11 13:58

#include <math.h>
#include <glog/logging.h>
#include "model/sgd_trainer.h"

namespace lapis {

void SGDTrainer::Init(const TrainerProto &trainer_proto) {
  Trainer::Init(trainer_proto);
  // take over the control for this pointer
  sgd_proto_ = trainer_proto.sgd();
  phase_ = TrainPhase::kTest;
}

SGDTrainer::~SGDTrainer() {
  sgd_proto_.Clear();
}

void SGDTrainer::BackPropagation(Net *net, const int step) {
  std::vector<Layer *> layers = net->Layers();
  std::vector<Edge *> edges = net->Edges();
  std::vector<Param *> params = net->Params();
  // get newest parameters for layers and edges
  model_controller_.GetParams(params);
  for (auto layer : layers)
    layer->Forward();
  for (auto layer = layers.rbegin(); layer != layers.rend(); layer++)
    (*layer)->Backward();
  UpdateHyperParams(step_);
  for (auto edge : edges) {
    edge->ComputeParamUpdates(this);
  }
  for (auto layer : layers) {
    layer->ComputeParamUpdates(this);
  }
  // update parameters either locally or distributedly depending on the
  // system (single machine or a cluster)
  model_controller_.UpdateParams(params);
}

void SGDTrainer::Train(Net *net, const int step) {
  if (phase_ != TrainPhase::kTrain) {
    for (auto layer : net->Layers()) {
      layer->Setup(sgd_proto_.train_batchsize(),
                   TrainAlgorithm::kBackPropagation,
                   train_data_);
    }
  }
  BackPropagation(net, step);
}

void SGDTrainer::Validate(Net *net) {
  if (phase_ != TrainPhase::kValidation) {
    for (auto layer : net->Layers()) {
      layer->Setup(sgd_proto_.validation_batchsize(),
                   TrainAlgorithm::kBackPropagation,
                   validation_data_);
    }
  }
  // TODO(wangwei) forward only
}

void SGDTrainer::Test(Net *net) {
  if (phase_ != TrainPhase::kTest) {
    for (auto layer : net->Layers()) {
      layer->Setup(sgd_proto_.test_batchsize(),
                   TrainAlgorithm::kBackPropagation,
                   test_data_);
    }
  }
  // TODO(wangwei) forward only
}

void SGDTrainer::ToProto(TrainerProto *proto) {
  Trainer::ToProto(proto);
  proto->set_allocated_sgd(&sgd_proto_);
}

bool SGDTrainer::HasFinished(int step) {
  if (step >= sgd_proto_.total_steps())
    return true;
  else
    return false;
}

float SGDTrainer::UpdateHyperParam(int step, SGDProto_ChangeProto change,
                                   int change_steps,
                                   float base_val, float final_val) {
  float ret = 0.;
  switch (change) {
  case SGDProto_ChangeProto_FIXED: {
    ret = base_val;
    break;
  }
  case SGDProto_ChangeProto_LINEAR: {
    float r = step * 1.0  / change_steps;
    ret = (1.0 - r) * base_val + r * final_val;
    break;
  }
  case SGDProto_ChangeProto_EXPONENTIAL: {
    CHECK_EQ(base_val, 2 * final_val) << "final value should be the half\n";
    ret = base_val / pow(2, step_ * 1. / change_steps);
    break;
  }
  case SGDProto_ChangeProto_INVERSE_T: {
    CHECK_EQ(base_val, 2 * final_val) << "final value should be the half\n";
    ret = base_val / (1. + step_ * 1. / change_steps);
    break;
  }
  default: {
    LOG(INFO) << "Wrong hyper-parameter update method\n";
  }
  }
  return ret;
}

void SGDTrainer::UpdateHyperParams(int step) {
  learning_rate_ = UpdateHyperParam(step, sgd_proto_.learning_rate_change(),
                                    sgd_proto_.learning_rate_change_steps(),
                                    sgd_proto_.base_learning_rate(),
                                    sgd_proto_.final_learning_rate());
  momentum_ = UpdateHyperParam(step, sgd_proto_.momentum_change(),
                               sgd_proto_.momentum_change_steps(),
                               sgd_proto_.base_momentum(),
                               sgd_proto_.final_momentum());
  weight_decay_ = UpdateHyperParam(step, sgd_proto_.weight_decay_change(),
                                   sgd_proto_.weight_decay_change_steps(),
                                   sgd_proto_.base_weight_decay(),
                                   sgd_proto_.final_weight_decay());
}
}  // namespace lapis
