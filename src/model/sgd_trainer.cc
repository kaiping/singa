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
  // model_controller_->Get(params);
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
  for (auto* param: params)
    param.mutable_content()+=param.history();
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

float UpdateHyperParam(int step, SGDProto::ChangeProto change,
                                   int change_steps, float a,
                                   float b) {
  float ret = 0., r=0.;
  switch (change) {
    case SGDProto::kFixed:
      ret = a;
      break;
    case SGDProto::kLinear:
      // a is init, b is the final
      r = step * 1.0  / change_steps;
      ret = (1.0 - r) * a + r * b;
      break;
    case SGDProto::kExponential:
      // a is init, b is the final, from convnet
      CHECK_EQ(a, 2 * b) << "final value should be the half";
      ret = a / pow(2, step * 1. / change_steps);
      break;
    case SGDProto::kInverse_t:
      // a is init, b is the final, from convnet
      CHECK_EQ(a, 2 * b) << "final value should be the half";
      ret = a / (1. + step * 1. / b);
      break;
    case SGDProto::kStep:
      // a is the base learning rate, b is gamma, from caffe
      ret=a*pow(b, step/change_steps);
    default:
      LOG(INFO) << "Wrong hyper-parameter update method";
  }
  return ret;
}

void SGDTrainer::UpdateHyperParams(int step) {
  learning_rate_ = UpdateHyperParam(step, sgd_proto_.learning_rate_change(),
                                    sgd_proto_.learning_rate_change_steps(),
                                    sgd_proto_.base_learning_rate(),
                                    sgd_proto_.learning_rate_x());
  momentum_ = UpdateHyperParam(step, sgd_proto_.momentum_change(),
                               sgd_proto_.momentum_change_steps(),
                               sgd_proto_.base_momentum(),
                               sgd_proto_.momentum_x());
  weight_decay_ = UpdateHyperParam(step, sgd_proto_.weight_decay_change(),
                                   sgd_proto_.weight_decay_change_steps(),
                                   sgd_proto_.base_weight_decay(),
                                   sgd_proto_.weight_decay_x());
}
}  // namespace lapis
