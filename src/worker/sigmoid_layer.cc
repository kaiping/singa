// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-10 22:52

#include <worker/sigmoid_layer.h"

namespace lapis {
void SigmoidLayer::Setup(const LayerProto& layer_proto,
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

  CHECK_EQ(params.size(), 0) << "Inner product layer should have no params\n";
  Blob* blob = out_edges_[0].Blob();
  new (&out_) MapArrayType(blob->content(), blob->num(), blob->width());
  new (&out_grad_) MapArrayType(blob->grad(), blob->num(), blob->width());
  blob = in_edges_[0].Blob();
  new (&in_) MapArrayType(blob->content(), blob->num(), blob->width());
  new (&in_grad_) MapArrayType(blob->grad(), blob->num(), blob->width());
}

// out=1/(1+exp(-in))
void SigmoidLayer::Forward() {
  out_ = (-in_).exp();
  out_ = 1 / (1 + out_);
}

// in_grad_ = out_grad_ * sigmoid'(in_)=out_grad*out_*(1-out_)
void SigmoidLayer::Backward() {
  in_grad_.noalias() = out_grad * out_ * (1 - out_);
}

void SigmoidLayer::ComputeParamUpdates(const SGD& sgd) {
  // sigmoid layer has no params
}

}  // namespace lapis
