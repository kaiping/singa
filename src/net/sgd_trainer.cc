// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-11 13:58

#include <math.h>
#include <glog/logging.h>
#include "net/data_layer.h"
#include "net/sgd_trainer.h"
#include "net/prediction_layer.h"

namespace lapis {

void SGDTrainer::Init(const TrainerProto &proto, ModelController *mc) {
  Trainer::Init(proto,mc);
  // take over the control for this pointer
  sgd_proto_ = proto.sgd();
}

SGDTrainer::~SGDTrainer() {
  sgd_proto_.Clear();
}

void SGDTrainer::TrainOneBatch(Net* net, Performance* perf) {
  perf->set_prefix("train");
  std::vector<Layer *> layers = net->layers();
  std::vector<Edge *> edges = net->edges();

  Timer timer;
  char buf[1024];
  int len=0;
  for (auto* layer : layers){
    timer.reset();
    if(layer->HasInput()){
      // TODO(wangwei) Error has not implemented mc.GetData.
      auto dlayer=dynamic_cast<DataLayer*>(layer);
      Blob& blob=dlayer->feature(nullptr);
      // TODO(wangwei) remove this outside to worker run();
      // worker passes map: data source name->Blob
      model_controller_->GetData(dlayer->store_id(), &blob);
    }
    layer->Forward();
    sprintf(buf+len, "%4.2f ", timer.elapsed());
    len=strlen(buf);
  }
  perf->MergeFrom(dynamic_cast<SoftmaxPredictionLayer*>(layers.back())->CalcPerf(true, false));
  for (auto layer = layers.rbegin(); layer != layers.rend(); layer++){
    timer.reset();
    (*layer)->Backward();
    sprintf(buf+len, "%4.2f ", timer.elapsed());
    len=strlen(buf);
  }
  UpdateHyperParams(step_);
  for (auto edge : edges)
    edge->ComputeParamUpdates(this);
  for (auto layer : layers)
    layer->ComputeParamUpdates(this);
  std::string timestr(buf);
  VLOG(1)<<timestr;
  IncStep();
}

std::string SGDTrainer::FormatTime(const std::vector<Edge *> &edges) {
  char buf[1024];
  sprintf(buf, "step:%4d ", step_);
  int count=step_+1;
  int strl=0;
  for(auto edge:edges) {
    char prefix=edge->PrefixAbbrev();
    sprintf(buf+strl, "%cf:%4.2f, %cb:%4.2f, ", prefix, edge->forward_time()/count,
        prefix,edge->backward_time()/count);
    strl=strlen(buf);
  }
  std::string s(buf);
  return s;
}

void SGDTrainer::Validate(Net *net, Performance* perf, int nbatches) {
  perf->set_prefix("val");
  std::vector<Layer *> layers = net->layers();
  std::vector<Param *> params = net->params();
  // get newest parameters for layers and edges
  float loss=0.0f, precision=0.0f;
  for(int k=0;k<nbatches;k++){
    for (auto* layer : layers){
      if(layer->HasInput()){
        // TODO(wangwei) Error has not implemented mc.GetData.
        auto dlayer=dynamic_cast<DataLayer*>(layer);
        Blob& blob=dlayer->feature(nullptr);
        VLOG(3)<<"getting data..";
        model_controller_->GetData(dlayer->store_id(), &blob);
      }
      layer->Forward();
    }
    Performance p=dynamic_cast<SoftmaxPredictionLayer*>(layers.back())->CalcPerf();
    loss+=p.loss();
    precision+=p.precision();
  }
  perf->set_loss(loss/nbatches);
  perf->set_precision(precision/nbatches);
}

void SGDTrainer::Test(Net *net, Performance* perf, int nbatches) {
  /*
  if (phase != Phase::kTest) {
    net->Setup(kAllocData, test_data_shapes_);
    phase = Phase::kTest;
  }
   TODO(wangwei) forward through all layers to get the loss
     for(int i=0;i<test_data_[0].size()/sgd_proto_.test_batchsize();i++)
     Forward();
     */
}

void SGDTrainer::ToProto(TrainerProto *proto) {
  //Trainer::ToProto(proto);
  //proto->set_allocated_sgd(&sgd_proto_);
}

bool SGDTrainer::HasFinished() {
  if (step_ >= sgd_proto_.total_steps())
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
  learning_rate_/=GlobalContext::Get()->num_processes()-1;
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
