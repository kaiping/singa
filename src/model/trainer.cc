// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-14 14:28

#include <google/protobuf/repeated_field.h>
#include <vector>
#include "proto/lapis.pb.h"
#include "model/trainer.h"


namespace lapis {

void InitDataSource(
    const ::google::protobuf::RepeatedPtrField<DataSourceProto> &protos,
    std::vector<DataSource *> *sources) {
  for (auto &proto : protos) {
    DataSource* ds=new DataSource(proto);
    sources->push_back(ds);
  }
}

void Trainer::Init(const TrainerProto &trainer_proto) {
  //! if step_>0, then the trainer is restored from a checkpoint
  step_ = trainer_proto.checkpoint_step();

  checkpoint_after_steps_ = trainer_proto.checkpoint_after_steps();
  checkpoint_every_steps_ = trainer_proto.checkpoint_every_steps();
  //! last checkpoint step
  checkpoint_step_ = trainer_proto.checkpoint_step();
  checkpoint_prefix_ = trainer_proto.checkpoint_prefix();

  display_after_steps_ = trainer_proto.display_after_steps();
  display_every_steps_ = trainer_proto.display_every_steps();
  display_prefix_ = trainer_proto.display_prefix();

  InitDataSource(trainer_proto.train_data(), &train_data_);
  InitDataSource(trainer_proto.validation_data(), &validation_data_);
  InitDataSource(trainer_proto.test_data(), &test_data_);

  perf_prefix_ = trainer_proto.perf_prefix();
}

void Trainer::ToProto(TrainerProto *proto) {
  proto->set_checkpoint_after_steps(checkpoint_after_steps_);
  proto->set_checkpoint_every_steps(checkpoint_every_steps_);
  proto->set_checkpoint_step(checkpoint_step_);
  proto->set_checkpoint_prefix(checkpoint_prefix_);

  proto->set_display_after_steps(display_after_steps_);
  proto->set_display_every_steps(display_every_steps_);
  proto->set_display_prefix(display_prefix_);

  proto->clear_train_data();
  for (DataSource *ds : train_data_) {
    DataSourceProto *ds_proto = proto->add_train_data();
    ds->ToProto(ds_proto);
  }
  proto->clear_validation_data();
  for (DataSource *ds : validation_data_) {
    DataSourceProto *ds_proto = proto->add_validation_data();
    ds->ToProto(ds_proto);
  }
  proto->clear_test_data();
  for (DataSource *ds : test_data_) {
    DataSourceProto *ds_proto = proto->add_test_data();
    ds->ToProto(ds_proto);
  }
}
inline const bool Trainer::CheckpointNow(const int step) {
  if (checkpoint_after_steps_ > 0 && step >= checkpoint_after_steps_) {
    if ((step - checkpoint_every_steps_) % checkpoint_every_steps_ == 0)
      return true;
  }
  return false;
}
inline void Trainer::IncStep() {
  step_++;
}
inline const int Trainer::Step() {
  return step_;
}


}  // namespace lapis
