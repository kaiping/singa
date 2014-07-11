// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-11 13:58

#include "worker/sgd.h"

namespace lapis {

SGD::SGD(const SGDProto& sgd_proto) {
  sgd_proto = sgd_proto;
}

void SGD::IncStep() {
  step_++;
}

bool SGD::Finished() {
  if (step_ < total_steps_) {
    return false;
  } else {
    return true;
  }
}

float updateField(SGDProto_ChangeProto change, int change_steps,
                  float base_val, float final_val) {
  float ret = 0.;
  switch(change) {
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
      ret = base_val / pwer(2, step_ * 1. / change_steps);
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

void SGD::UpdateHyperParams() {
  learning_rate_ = updateField(sgd_proto_.learning_rate_change(),
                               sgd_proto_.learning_rate_change_steps(),
                               sgd_proto_.base_learning_rate(),
                               sgd_proto_.final_learning_rate());
  momentum_ = updateField(sgd_proto_.momentum_change(),
                          sgd_proto_.momentum_change_steps(),
                          sgd_proto_.base_momentum(),
                          sgd_proto_.final_momentum());
  weight_decay_ = updateField(sgd_proto_.weight_decay_change(),
                              sgd_proto_.weight_decay_change_steps(),
                              sgd_proto_.base_weight_decay(),
                              sgd_proto_.final_weight_decay());
}
}  // namespace lapis
