// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-14 14:28

#include <google/protobuf/repeated_field.h>
#include <glog/logging.h>
#include <vector>

#include "proto/model.pb.h"
#include "net/trainer.h"

namespace lapis {

void Solver(const TrainerProto &proto) {
  //! if step_>0, then the trainer is restored from a checkpoint
  step_ = proto.checkpoint_step();
  checkpoint_after_steps_ = proto.checkpoint_after_steps();
  checkpoint_every_steps_ = proto.checkpoint_every_steps();
  //! last checkpoint step
  checkpoint_step_ = proto.checkpoint_step();
  checkpoint_prefix_ = proto.checkpoint_prefix();
  display_after_steps_ = proto.display_after_steps();
  display_every_steps_ = proto.display_every_steps();
  display_prefix_ = proto.display_prefix();
  validate_after_steps_ = proto.validate_after_steps();
  validate_every_steps_ = proto.validate_every_steps();

  perf_prefix_ = proto.perf_prefix();
  do_train_ = proto.do_train();
  do_test_ = proto.do_test();
}

void Solver::ToProto(TrainerProto *proto) {
  proto->set_checkpoint_after_steps(checkpoint_after_steps_);
  proto->set_checkpoint_every_steps(checkpoint_every_steps_);
  proto->set_checkpoint_step(checkpoint_step_);
  proto->set_checkpoint_prefix(checkpoint_prefix_);
  proto->set_display_after_steps(display_after_steps_);
  proto->set_display_every_steps(display_every_steps_);
  proto->set_display_prefix(display_prefix_);
}

void Solver::Forward(Net* net) {
  for (auto* layer : net->layers()){
    layer->Forward();
  }
}

void Solver::Backward(Net* net) {
  std::vector<Layer *> layers = net->layers();
  for (auto layer = layers.rbegin(); layer != layers.rend(); layer++)
    (*layer)->Backward();
}

Performance Solver::TrainOneBatch(Net *net){
  Forward(net);
  Performance perf=net->output_layer()->CalcPerf(true, false);
  Backward(net);
  IncStep();
  return perf;
}

Performance Solver::ValidateOneBatch(Net *net){
  Forward(net);
  return net->output_layer()->CalcPerf(true, true);
}

//Performance Solver::Test(Net *net) { }

bool Solver::HasFinished() {
  if (step_ >= total_steps_)
    return true;
  else
    return false;
}

Trainer::~Trainer() {
}
}  // namespace lapis
