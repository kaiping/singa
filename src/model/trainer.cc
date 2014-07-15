// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-14 14:28

#include <google/protobuf/repeated_field.h>
#include <vector>
#include "proto/lapis.pb.h"
#include "model/trainer.h"

using google::protobuf::RepeatedPtrFeild;
using std::vector;

namespace lapis {

void InitDataSource(const RepeatedPtrFeild<DataSourceProto> &protos,
                    vector<DataSource *> *sources) {
  for (auto &proto : protos) {
    auto *data = DataSourceFactory::Instance()->Create(data_source.type());
    data->Init(proto);
    sources.push_back(data);
  }
}

void Trainer::Init(const TrainerProto &trainer_proto) {
  //! if step_>0, then the trainer is restored from a checkpoint
  step_ = trainer_proto.step();

  checkpoint_after_steps_ = trainer_proto.checkpoint_after_steps();
  checkpoint_every_steps_ = trainer_proto.checkpoint_every_steps();
  //! last checkpoint step
  checkpoint_step_ = trainer_proto.checkpoint_step();
  checkpoint_prefix_ = trainer_proto.checkpoint_prefix();

  display_after_steps_ = trainer_proto.display_after_steps();
  display_every_steps = trainer_proto.display_every_steps();
  display_prefix_ = trainer_proto.display_prefix();

  InitDataSource(trainer_proto.train_data(), train_data_);
  InitDataSource(trainer_proto.validation_data(), validation_data_);
  InitDataSource(trainer_proto.test_data(), test_data_);

  perf_prefix_ = trainer_proto.perf_prefix();
}

void Trainer::ToProto(TrainerProto *proto) {
  proto->set_step(step_);

  proto->set_checkpoint_after_steps(checkpoint_after_steps_);
  proto->set_checkpoint_every_steps(checkpoint_every_steps_);
  proto->set_checkpoint_steps(checkpoint_steps_);
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
}  // namespace lapis
