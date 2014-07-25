// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-15 18:46

#include <glog/logging.h>
#include "utils/lapis.h"
#include "model/inner_product_edge.h"

namespace lapis {

const std::string InnerProductEdge::kInnerProductEdge = "InnerProduct";
void InnerProductEdge::Init(const EdgeProto &proto,
                            const std::map<std::string, Layer *> &layer_map) {
  Edge::Init(proto, layer_map);
  param_proto_=proto.param();
  num_output_=proto.num_output();
}

void InnerProductEdge::Setup(bool set_param) {
  if(set_param) {
    CHECK(param_proto_.size()<=2);
    for (auto proto : param_proto_) {
      if(proto.name()=="weight") {
        proto.clear_shape();
        proto.add_shape(bottom_->feature(this)->record_length());
        proto.add_shape(num_output_);
        weight_.Init(proto);
        params_.push_back(&weight_);
      } else if (proto.name()=="bias") {
        proto.clear_shape();
        proto.add_shape(num_output_);
        bias_.Init(proto);
        params_.push_back(&bias_);
      }
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

void InnerProductEdge::Forward(const Blob *src, Blob *dest, bool overwrite) {
  MMat fea(src->mutable_data(), src->height(), src->width());
  MMat act(dest->mutable_data(), dest->width(), dest->width());
  MMat weight(weight_.mutable_content(), weight_.height(), weight_.width());
  MVec bias(bias_.mutable_content(), bias_.length());
  if (overwrite)
    act.noalias() = (fea * weight).rowwise() + bias;
  else
    act += (fea * weight).rowwise() + bias;
}

void InnerProductEdge::Backward(const Blob *src_grad, const Blob *dest_fea,
                                Blob *dest_grad, bool overwrite) {
  MMat act_grad(src_grad->mutable_data(), src_grad->height(),
                         src_grad->width());
  MMat fea(dest_fea->mutable_data(), dest_fea->width(),
                    dest_fea->width());
  MMat weight_grad(weight_.mutable_gradient(), weight_.height(), weight_.width());
  MMat weight(weight_.mutable_content(), weight_.height(), weight_.width());
  MVec bias_grad(bias_.mutable_gradient(), bias_.length());
  weight_grad.noalias() = fea.transpose() * act_grad;
  bias_grad = act_grad.colwise().sum();
  // if dest_grad is nullptr, then we only compute gradients for parameters
  // this may happen when the lower layer is DataLayer
  if (dest_grad != nullptr) {
    MMat fea_grad(dest_grad->mutable_data(), dest_grad->height(),
                           dest_grad->width());
    if (overwrite)
      fea_grad.noalias() = act_grad * weight.transpose();
    else
      fea_grad += act_grad * weight.transpose();
  }
}

void InnerProductEdge::SetupTopBlob(Blob* blob) {
  Blob* b=bottom_->feature(this);
  blob->Reshape(b->num(),1, b->record_length(), num_output_);
}
}  // namespace lapis
