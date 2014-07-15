// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-15 18:46

#incldue "model/inner_product_edge.h"

namespace lapis {
void InnerProductEdge::Init(const EdgeProto &edge_proto,
                            const map<string, Edge *> &edge_map) {
  Edge::Init(edge_proto);
  CHECK_EQ(edge_proto.param().size(),
           2) << "InnerProductEdge should have two parameters\n";
  //! match the parameter based on shape
  auto &param0 = edge_proto.param().Get(0);
  auto &param1 = edge_proto.param().Get(1);
  if (shape0.size() == 2) {
    weight.Init(param0);
    bias.Init(param1);
  } else {
    weight.Init(param1);
    bias.Init(param0);
  }
}

void InnerProductEdge::ToProto(EdgeProto *edge_proto) {
  Edge::ToProto(edge_proto);
  edge_proto.set_type(kInnerProductEdge);
  ParamProto *weight_proto = edge_proto->add_param();
  weight.ToProto(weight_proto);
  ParamProto *bias_proto = edge_proto->add_param();
  bias.ToProto(bias_proto);
}

void InnerProductEdge::Forward(const Blob *src, Blob *dest, bool overwrite) {
  MapMatrixType fea(src->Content(), src->Rows(), src->Cols());
  MapMatrixType act(dest->Content(), dest->Rows(), dest->Cols());
  MapMatrixType weight(weight_.Content(), weight_.Rows(), weight_.Cols());
  MapVectorType bias(bias_.Content(), bias_.Length());

  if (overwrite)
    act.noalias() = (fea * weight).rowwise() + bias;
  else
    act += (fea * weight).rowwise() + bias;
}

void InnerProductEdge::Backward(const Blob *src_grad, const Blob *des_fea,
                                Blob *des_grad, bool overwrite) {
  MapMatrixType act_grad(src_grad->Content(), src_grad->Rows(),
                         src_grad->Cols());
  MapMatrixType fea_grad(dest_grad->Content(), dest_grad->Rows(),
                         dest_grad->Cols());
  MapMatrixType fea(dest->Content(), dest->Rows(), dest->Cols());
  MapMatrixType weight_grad(weight_grad_.Content(), weight_grad_.Rows(),
                            weight_grad_.Cols());
  MapMatrixType weight(weight__.Content(), weight_.Rows(), weight_.Cols());
  MapVectorType bias_grad(bias_grad_.Content(), bias_grad_.Length());

  weight_grad.noalias() = fea.transpose() * act_grad;
  bias_grad = act_grad.colwise().sum();
  if (overwrite)
    fea_grad.noalias() = act_grad * weight.transpose();
  else
    fea_grad += act_grad * weight.transpose();
}

}  // namespace lapis
