// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-11 13:58

#include <math.h>
#include <glog/logging.h>
#include "net/data_layer.h"

#include "net/sgd_trainer.h"

namespace lapis {

void SGDTrainer::Init(const TrainerProto &proto, ModelController *mc) {
  Trainer::Init(proto,mc);
  // take over the control for this pointer
  sgd_proto_ = proto.sgd();
}

SGDTrainer::~SGDTrainer() {
  sgd_proto_.Clear();
}

void SGDTrainer::BackPropagation(const int step, Net* net) {
  std::vector<Layer *> layers = net->layers();
  std::vector<Edge *> edges = net->edges();
  std::vector<Param *> params = net->params();
  // get newest parameters for layers and edges
  VLOG(3)<<"before get params from distributed mem";
  model_controller_->Get(params);
  VLOG(3)<<"after get params from distributed mem";

  for (auto* layer : layers)
    layer->Forward();
  for (auto layer = layers.rbegin(); layer != layers.rend(); layer++)
    (*layer)->Backward();
  UpdateHyperParams(step_);
  for (auto edge : edges)
    edge->ComputeParamUpdates(this);
  for (auto layer : layers)
    layer->ComputeParamUpdates(this);

  VLOG(3)<<"before update params from distributed mem";
  model_controller_->Update(params);
  VLOG(3)<<"after update params from distributed mem";
  // update parameters either locally or distributedly depending on the
  // system (single machine or a cluster)
  /*
  for (auto *param : params) {
    Tensor1 p(param->mutable_content().dptr, Shape1(param->length()));
    const Tensor1 h(param->history().dptr, Shape1(param->length()));
    p += h;
  }
  */
}

void SGDTrainer::Train(const int step, Net *net, const char flag) {
  if (phase != Phase::kTrain) {
    char local_flag=step==0?flag: kAllocData;
    net->Setup(sgd_proto_.train_batchsize(),
               local_flag,
               train_data_);
    phase = Phase::kTrain;
    VLOG(1)<<"Total mem allocated for Blobs in training is "
      <<Blob::MSize()<<" megabytes";
  }
  BackPropagation(step, net);
}

void SGDTrainer::Validate(Net *net) {
  if (phase != Phase::kValidation) {
    net->Setup(sgd_proto_.validation_batchsize(),
               kAllocData,
               validation_data_);
    phase = Phase::kValidation;
  }
  /* TODO(wangwei) forward through all layers to get the loss
     for(int i=0;i<test_data_[0].size()/sgd_proto_.test_batchsize();i++)
     Forward();
     */
}

void SGDTrainer::Test(Net *net) {
  if (phase != Phase::kTest) {
    net->Setup(sgd_proto_.test_batchsize(),
               kAllocData,
               test_data_);
    phase = Phase::kTest;
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
  float ret = 0., r = 0.;
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
      ret = a * pow(b, step / change_steps);
      break;
    default:
      LOG(ERROR) << "Wrong hyper-parameter update method";
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
