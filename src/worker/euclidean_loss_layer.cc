// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-11 11:04

#include "worker/euclidean_loss_layer.h"

namespace lapis {

void EuclideanLossLayer::Setup(const LayerProto& layer_proto,
                               std::map<string, Edge*>* edges) {
  CHECK_EQ(in_edges_.size(), 2) << "Eucliden loss layer should have 2 in edges"
                                << "now it has " << in_edges_.size() << "\n";

  Blob* blob = nullptr;
  blob = in_edges_[0].Blob();
  new (&predict_) MapMatrixType(blob->content(), blob->num(), blob->width());
  new (&grad_) MapMatrixType(blob->grad(), blob->num(), blob->width());

  blob = in_edges_[1].Blob();
  new (&label_) MapMatrixType(blob->content(), blob->num(), blob->width());
}

void EuclideanLossLayer::Forward() {
  loss_ = (predict_ - label_).norm() / predict_.rows();
}

void EuclideanLossLayer::Backward() {
  grad_ = (predict_ - label_) / (predict_.rows() * 2.0);
}

void EuclideanLossLayer::ComputeParamUpdates(const SGD& sgd) {
  // euclidean loss layer has no params
}
}  // namespace lapis
