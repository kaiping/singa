// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-14 14:28

#include <google/protobuf/repeated_field.h>
#include <glog/logging.h>
#include <vector>

#include "proto/model.pb.h"
#include "net/solver.h"

namespace lapis {
int Solver::phase=kTrain;
Solver::Solver(const SolverProto &proto) {
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
  validation_after_steps_ = proto.validation_after_steps();
  validation_every_steps_ = proto.validation_every_steps();

  perf_prefix_ = proto.perf_prefix();
  train_steps_=proto.train_steps();
  validation_steps_=proto.validation_steps();
}

void Solver::ToProto(SolverProto *proto) {
  proto->set_checkpoint_after_steps(checkpoint_after_steps_);
  proto->set_checkpoint_every_steps(checkpoint_every_steps_);
  proto->set_checkpoint_step(checkpoint_step_);
  proto->set_checkpoint_prefix(checkpoint_prefix_);
  proto->set_display_after_steps(display_after_steps_);
  proto->set_display_every_steps(display_every_steps_);
  proto->set_display_prefix(display_prefix_);
}

Performance Solver::TrainOneBatch(Net *net){
  net->Forward();
  Performance perf=net->output_layer(0)->CalcPerf(true, false);
  net->Backward();
  IncStep();
  return perf;
}

Performance Solver::ValidateOneBatch(Net *net){
  net->Forward();
  return net->output_layer(0)->CalcPerf(true, true);
}

//Performance Solver::Test(Net *net) { }

bool Solver::HasFinished(){
  if (step_ >= train_steps_)
    return true;
  else
    return false;
}

}  // namespace lapis
