// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-15 20:15

#include "model/eucldiean_loss_edge.h"

namespace lapis {
void EuclideanLossEdge::Init(const EdgeProto &edge_proto) {
  Edge::Init(edge_proto);
}
void EuclideanLossEdge::ToProto(EdgeProto *edge_proto) {
  Edge::ToProto(edge_proto);
}
void EuclideanLossEdge::Forward(const Blob *src, Blob *dest, bool overwrite) {
  LOG(INFO) << "Forward() of EuclideanLossEdge should not be called\n";
}
void EuclideanLossEdge::Backward(const Blob *src_grad, const Blob *dest_fea,
                                 Blob *dest_grad) {
  MapMatrixType predict(dest_fea->Content(), dest_fea->Rows(), dest->Cols());
  //! here src_grad is actually the input data/label from DataLayer
  MapMatrixType data(src_grad->Content(), src_grad->Rows(), src_grad->Cols());
  MapMatrixType fea_grad(dest_grad->Content(), dest_grad->Rows(),
                         dest->Cols());
  fea_grad = (predict - data) / predict.Rows();
}
}  // namespace lapis
