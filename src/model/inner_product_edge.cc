// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-15 18:46

#include <glog/logging.h>
#include "model/lapis.h"
#include "model/inner_product_edge.h"
#include "utils/common.h"


namespace lapis {

const std::string InnerProductEdge::kInnerProductEdge = "InnerProduct";
void InnerProductEdge::Init(const EdgeProto &proto,
                            const std::map<std::string, Layer *> &layer_map) {
  Edge::Init(proto, layer_map);
  param_proto_ = proto.param();
  num_output_ = proto.num_output();
  params_.clear();
  params_.push_back(&weight_);
  params_.push_back(&bias_);
}

void InnerProductEdge::Setup(const char flag) {
  const Blob &b = bottom_->feature(this);
  num_ = b.num();
  num_input_ = b.length() / num_;
  CHECK(param_proto_.size() <= 2);
  for (auto proto : param_proto_) {
    if (proto.name() == "weight") {
      proto.clear_shape();
      proto.add_shape(num_input_);
      proto.add_shape(num_output_);
      weight_.Init(proto, flag);
      params_.push_back(&weight_);
    } else if (proto.name() == "bias") {
      proto.clear_shape();
      proto.add_shape(num_output_);
      bias_.Init(proto, flag);
      params_.push_back(&bias_);
    }
  }
}

void InnerProductEdge::ToProto(EdgeProto *edge_proto) {
  Edge::ToProto(edge_proto);
  edge_proto->set_type(kInnerProductEdge);
  ParamProto *weight_proto = edge_proto->add_param();
  weight_.ToProto(weight_proto);
  ParamProto *bias_proto = edge_proto->add_param();
  bias_.ToProto(bias_proto);
}

void InnerProductEdge::Forward(const Blob &src, Blob *dest, bool overwrite) {
  VLOG(3)<<name_;
  const Tensor2 src2(src.dptr, Shape2(num_, num_input_));
  Tensor2 dest2(dest->dptr, Shape2(num_, num_output_));
  Tensor2 weight(weight_.content().dptr, Shape2(num_input_, num_output_));
  Tensor1 bias(bias_.content().dptr, Shape1(num_output_));
  // Note, can not join the dot and repmat operations! the resulted expr is not
  // defined in mshadow
  if (overwrite)
    dest2 = dot(src2, weight);
  else
    dest2 += dot(src2, weight);
  dest2+= mshadow::expr::repmat(bias, num_);
}

void InnerProductEdge::Backward(const Blob &src_fea, const Blob &src_grad,
                                const Blob &dest_fea, Blob *dest_grad, bool overwrite) {
  VLOG(3)<<name_;
  Tensor2 dest_fea2(dest_fea.dptr, Shape2(num_,num_input_));
  Tensor2 src_grad2(src_grad.dptr, Shape2(num_,num_output_));
  Tensor2 weight_grad(weight_.mutable_gradient().dptr, Shape2(num_input_,num_output_));
  weight_grad = dot(dest_fea2.T(), src_grad2);

  Tensor1 bias_grad(bias_.mutable_gradient().dptr, Shape1(num_output_));
  bias_grad = mshadow::expr::sum_rows(src_grad2);
  // if dest_grad is nullptr, then we only compute gradients for parameters
  // this may happen when the lower layer is DataLayer
  if (dest_grad != nullptr) {
    const Tensor2 weight (weight_.content().dptr, Shape2(num_input_, num_output_));;
    Tensor2 dest_grad2(dest_grad->dptr, Shape2(num_,num_input_));
    if (overwrite)
      dest_grad2 = dot(src_grad2, weight.T());
    else
      dest_grad2 = dot(src_grad2, weight.T());
  }
}

void InnerProductEdge::SetupTopBlob(const bool alloc, Blob *blob) {
  blob->Resize(num_, 1, 1, num_output_, alloc);
}
}  // namespace lapis
