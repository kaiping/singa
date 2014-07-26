// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-11 13:58

#include <math.h>
#include <glog/logging.h>
#include "model/sgd_trainer.h"

namespace lapis {

void SGDTrainer::Init(const TrainerProto &proto, ModelController *mc) {
  Trainer::Init(proto, mc);
  // take over the control for this pointer
  sgd_proto_ = proto.sgd();
  phase_ = Phase::kInit;
}

SGDTrainer::~SGDTrainer() {
  sgd_proto_.Clear();
}

void SGDTrainer::BackPropagation(Net *net, const int step) {
  std::vector<Layer *> layers = net->layers();
  std::vector<Edge *> edges = net->edges();
  std::vector<Param *> params = net->params();
  // get newest parameters for layers and edges
  model_controller_->Get(params);
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
  model_controller_->Update(params);
}

void SGDTrainer::Train(Net *net, const int step) {
  if (phase_ != Phase::kTrain) {
    for (auto *layer : net->layers()) {
      layer->Setup(sgd_proto_.train_batchsize(),
                   TrainerProto::kBackPropagation,
                   train_data_);
      for(auto* edge: layer->out_edges())
        edge->Setup(true);
    }
    phase_=Phase::kTrain;
  }
  BackPropagation(net, step);
}

void SGDTrainer::Validate(Net *net) {
  if (phase_ != Phase::kValidation) {
    for (auto *layer : net->layers()) {
      layer->Setup(sgd_proto_.validation_batchsize(),
                   TrainerProto::kBackPropagation,
                   validation_data_);
      for(auto* edge: layer->out_edges())
        edge->Setup(false);
    }
    phase_=Phase::kValidation;
  }
  /* TODO(wangwei) forward through all layers to get the loss
  for(int i=0;i<test_data_[0].size()/sgd_proto_.test_batchsize();i++)
    Forward();
  */
}

void SGDTrainer::Test(Net *net) {
  if (phase_ != Phase::kTest) {
    for (auto* layer : net->layers()) {
      layer->Setup(sgd_proto_.test_batchsize(),
                   TrainerProto::kBackPropagation,
                   test_data_);
      for(auto* edge: layer->out_edges())
        edge->Setup(false);
    }
    phase_=Phase::kTest;
  }
  /* TODO(wangwei) forward through all layers to get the loss
  for(int i=0;i<test_data_[0].size()/sgd_proto_.test_batchsize();i++)
    Forward();
  */
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

float SGDTrainer::UpdateHyperParam(int step, SGDProto::ChangeProto change,
                                   int change_steps, float base_val,
                                   float final_val) {
  float ret = 0., r=0.;
  switch (change) {
    case SGDProto::kFixed:
      ret = base_val;
      break;
    case SGDProto::kLinear:
      r = step * 1.0  / change_steps;
      ret = (1.0 - r) * base_val + r * final_val;
      break;
    case SGDProto::kExponential:
      CHECK_EQ(base_val, 2 * final_val) << "final value should be the half";
      ret = base_val / pow(2, step_ * 1. / change_steps);
      break;
    case SGDProto::kInverse_t:
      CHECK_EQ(base_val, 2 * final_val) << "final value should be the half";
      ret = base_val / (1. + step_ * 1. / change_steps);
      break;
    default:
      LOG(INFO) << "Wrong hyper-parameter update method";
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
