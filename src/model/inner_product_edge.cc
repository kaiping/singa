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
  const Shape4& shape=(bottom_->feature(this)).shape;
  num_=shape[3];
  num_input_=shape.Size()/num_;
  if(set_param) {
    CHECK(param_proto_.size()<=2);
    for (auto proto : param_proto_) {
      if(proto.name()=="weight") {
        proto.clear_shape();
        proto.add_shape(num_input_);
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

void InnerProductEdge::Forward(const Blob4 &src, Blob4 *dest, bool overwrite) {
  const Blob2 src2=reshape(src, Shape2(num_input_, num_));
  Blob2 dest2=reshape(*dest, Shape2(num_output_,num_));
  Blob2& weight=weight_.content();
  Blob1& bias=bias_.content();
  if (overwrite)
    dest2 = (src2*weight) + repmat<1>(bias, num_);
  else
    dest += (src2 * weight) +repmat<1>(bias, num_);;
}

void InnerProductEdge::Backward(const Blob4 &src_grad, const Blob4 &dest_fea,
                                Blob4 *dest_grad, bool overwrite) {
  Blob2 dest_fea2=reshape(dest_fea, Shape2(num_input_,num_));
  Blob2 src_grad2=reshape(src, Shape2(num_output_, num_));
  Blob1 bias_grad=bias_.mutable_gradient();
  Blob2& weight_grad=weight_.mutable_gradient();
  weight_grad=dot(dest_fea2.T(), src_grad2);
  bias_grad = src_grad2.sum_rows();
  // if dest_grad is nullptr, then we only compute gradients for parameters
  // this may happen when the lower layer is DataLayer
  if (dest_grad != nullptr) {
    const Blob2& weight=weight_.content();
    Blob2 dest_grad2=reshape(*dest_grad, Shape2(num_input_, num_));
    if (overwrite)
      dest_grad2 = src_grad2 * weight.T();
    else
      dest_grad2= src_grad2 * weight.T();
  }
}

void InnerProductEdge::SetupTopBlob(Blob4* blob) {
  blob->Resize(Shape4(num_output_,1,1,num_));
}
}  // namespace lapis
