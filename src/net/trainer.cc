// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-14 14:28

#include <google/protobuf/repeated_field.h>
#include <glog/logging.h>
#include <vector>

#include "proto/model.pb.h"
#include "net/trainer.h"

namespace lapis {

Phase Trainer::phase=Phase::kInit;

/*
void Trainer::InitDataSource(
  const ::google::protobuf::RepeatedPtrField<DataSourceProto> &protos,
  std::vector<DataSource *> *sources) {
  std::shared_ptr<std::vector<std::string>> filenames;
  for (auto &proto : protos) {
    DataSource *ds = DataSourceFactory::Instance()->Create(proto.type());
    ds->Init(proto);
    LOG(INFO)<<"Created datasource: "<<ds->name();
    // TODO(this is for imagenet dataset)
    filenames=ds->LoadData(filenames);
    sources->push_back(ds);
  }
}
*/

void Trainer::Init(const TrainerProto &proto , ModelController *mc) {
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

  model_controller_ = mc;
}

void Trainer::ToProto(TrainerProto *proto) {
  proto->set_checkpoint_after_steps(checkpoint_after_steps_);
  proto->set_checkpoint_every_steps(checkpoint_every_steps_);
  proto->set_checkpoint_step(checkpoint_step_);
  proto->set_checkpoint_prefix(checkpoint_prefix_);
  proto->set_display_after_steps(display_after_steps_);
  proto->set_display_every_steps(display_every_steps_);
  proto->set_display_prefix(display_prefix_);
  /*
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
  */
}

Trainer::~Trainer() {
}
}  // namespace lapis
