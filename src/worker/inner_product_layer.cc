// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-08 15:31

#include <glog/logging.h>

#include "worker/inner_product_layer.h"

namespace lapis {
void InnerProductLayer::setup(const LayerProto& layer_proto,
                              std::map<string, Edge*>* edges) {
  CHECK_EQ(in_edges_.size(), 1) << "more than one in edges "
                                << "for inner product layer\n"
  CHECK_EQ(out_edges_.size(), 1) << "more than one out edges "
                                 << "for inner product layer\n";
  if (out_edges_[0].blob() == nullptr) {
    Blob* blob = new Blob(in_edges_[0].data().num(), 0,
                          0, layer_proto.output_size());
    out_edges_[0].SetData(blob);
  }

  CHECK_EQ(params.size(), 2) << "Inner product layer has more than 2 params\n";
  Param * weight, *bias;
  if (params[0]->shape().size() == 2) {
    weight_param_ = &params[0];
    bias_param_ = &params[1];
  } else {
    weight_param_ = &params[0];
    bias_param_ = &params[1];
  }
  new (&weight_)MapMatrixType(weight_param_->content(),
                              weight_param_->shape[0],
                              weight_param_->shape[1]);
  new (&weight_grad_)MapMatrixType(weight_param_->grad(),
                                    weight_param_->shape[0],
                                    weight_param_->shape[1]);
  new (&weight_history_grad_)MapMatrixType(weight_param_->history_grad_(),
                                            weight_param_->shape[0],
                                            weight_param_->shape[1]);
  new (&bias_)MapVectorType(bias_param_->content(), bias_param_->shape[0]);
  new (&bias_grad_)MapVectorType(bias_param_->grad(), bias_param_->shape[0]);
  new (&bias_history_grad_)MapVectorType(bias_param_->history_grad_(),
                                         bias_param_->shape[0]);

  Blob* blob = out_edges_[0].Blob();
  new (&out_) MapMatrixType(blob->content(), blob->num(), blob->width());
  new (&out_grad_) MapMatrixType(blob->grad(), blob->num(), blob->width());
  blob = in_edges_[0].Blob();
  new (&in_) MapMatrixType(blob->content(), blob->num(), blob->width());
  new (&in_grad_) MapMatrixType(blob->grad(), blob->num(), blob->width());
}

void InnerProductLayer::Forward() {
  out_.noalias() = (in_ * weight_).rowwise() + bias_;
}

void InnerProductLayer::Backward() {
  // calc gradient of W and b
  weight_grad_.noalias() = in_.transpose() * out_grad_;
  bias_grad_ = out_grad_.colwise().sum();

  in_grad_.noalias() = out_grad_ * weight_.transpose();
}

void InnerProductLayer::ComputeParamUpdates(const SGD& sgd) {
  // compute updates for weight
  float momentum = sgd->momentum() * weight_param_.momentum();
  float learning_rate = sgd->learning_rate() * weight_param_.learning_rate();
  float weight_decay = sgd->weight_decay() * weight_param_.weight_decay();
  if (momentum > 0)
    weight_history_grad_ *= momentum;
  if (weight_decay > 0)
    weight_history_grad_ -= (weight_grad_ + weight_decay * weight_)
                            * learning_rate;
  else
    weight_history_grad_ -= weight_grad_ * learning_rate;

  // compute updates for bias
  momentum = sgd->momentum() * bias_param_.momentum();
  learning_rate = sgd->learning_rate() * bias_param_.learning_rate();
  weight_decay = sgd->weight_decay() * bias_param_.weight_decay();
  if (momentum > 0)
    bias_history_grad_ *= momentum;
  if (weight_decay > 0)
    bias_history_grad_ -= (bias_grad_ + bias_decay * bias_) * learning_rate;
  else
    bias_history_grad_ -= bias_grad_ * learning_rate;
}
}  // namespace lapis
