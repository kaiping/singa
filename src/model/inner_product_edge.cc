// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-15 18:46

#include <glog/logging.h>
#include "utils/lapis.h"
#include "model/inner_product_edge.h"

namespace lapis {

const std::string InnerProductEdge::kInnerProductEdge = "InnerProduct";
void InnerProductEdge::Init(const EdgeProto &edge_proto,
                            const std::map<std::string, Edge *> &edge_map) {
  Edge::Init(edge_proto);
  CHECK_EQ(edge_proto.param().size(),
           2) << "InnerProductEdge should have two parameters\n";
  //! match the parameter based on shape
  auto &param0 = edge_proto.param().Get(0);
  auto &param1 = edge_proto.param().Get(1);
  if (param0.shape().size() == 2) {
    weight_.Init(param0);
    bias_.Init(param1);
  } else {
    weight_.Init(param1);
    bias_.Init(param0);
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
  MapMatrixType fea((const_cast<Blob *>(src))->MutableContent(), src->Height(),
                    src->Width());
  MapMatrixType act(dest->MutableContent(), dest->Width(), dest->Width());
  MapMatrixType weight(weight_.MutableContent(),
                       weight_.Rows(), weight_.Cols());
  MapVectorType bias(bias_.MutableContent(), bias_.Length());

  if (overwrite)
    act.noalias() = (fea * weight).rowwise() + bias;
  else
    act += (fea * weight).rowwise() + bias;
}

void InnerProductEdge::Backward(const Blob *src_grad, const Blob *dest_fea,
                                Blob *dest_grad, bool overwrite) {
  MapMatrixType act_grad(src_grad->MutableContent(), src_grad->Height(),
                         src_grad->Width());
  MapMatrixType fea(dest_fea->MutableContent(), dest_fea->Width(),
                    dest_fea->Width());
  MapMatrixType weight_grad(weight_.MutableGradient(), weight_.Rows(),
                            weight_.Cols());
  MapMatrixType weight(weight_.MutableContent(), weight_.Rows(),
                       weight_.Cols());
  MapVectorType bias_grad(bias_.MutableGradient(), bias_.Length());

  weight_grad.noalias() = fea.transpose() * act_grad;
  bias_grad = act_grad.colwise().sum();
  // if dest_grad is nullptr, then we only compute gradients for parameters
  // this may happen when the lower layer is DataLayer
  if (dest_grad != nullptr) {
    MapMatrixType fea_grad(dest_grad->MutableContent(), dest_grad->Height(),
                           dest_grad->Width());
    if (overwrite)
      fea_grad.noalias() = act_grad * weight.transpose();
    else
      fea_grad += act_grad * weight.transpose();
  }
}

}  // namespace lapis
