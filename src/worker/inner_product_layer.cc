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
  if (params[0]->shape().size() == 2) {
    new (&weight_)MapMatrixType(params[0].content(),
                                params[0].shape[0],
                                params[0].shape[1]);
    new (&weight_grad_)MapMatrixType(params[0].grad(),
                                params[0].shape[0],
                                params[0].shape[1]);
    new (&bias_)MapVectorType(params[1].content(), params[1].shape[0]);
    new (&bias_grad_)MapVectorType(params[1].grad(), params[1].shape[0]);

  } else {
    new (&weight_)MapMatrixType(params[1].content(),
                                params[1].shape[0],
                                params[1].shape[1]);
    new (&weight_grad_)MapMatrixType(params[1].grad(),
                                params[1].shape[0],
                                params[1].shape[1]);
    new (&bias_)MapVectorType(params[0].content(), params[0].shape[0]);
    new (&bias_grad_)MapVectorType(params[0].grad(), params[0].shape[0]);
  }

  Blob* blob = out_edges_[0].Blob();
  new (&out_) MapMatrixType(blob->content(), blob->num(), blob->width());
  new (&out_grad_) MapMatrixType(blob->grad(), blob->num(), blob->width());
  blob = in_edges_[0].Blob();
  new (&in_) MapMatrixType(blob->content(), blob->num(), blob->width());
  new (&in_grad_) MapMatrixType(blob->grad(), blob->num(), blob->width());
}

void InnerProductLayer::forward() {
  out_.noalias() = (in_ * weight_).rowwise() + bias_;
}

void InnerProductLayer::backward() {
  // calc gradient of W and b
  weight_grad_.noalias() = in_.transpose() * out_grad_;
  bias_grad_ = out_grad_.colwise().sum();

  in_grad_.noalias() = out_grad_ * weight_.transpose();
}

}  // namespace lapis
