// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-15 20:15

#include <glog/logging.h>

#include "utils/lapis.h"
#include "model/euclidean_loss_edge.h"

namespace lapis {

const std::string kEuclideanLossEdge = "EuclideanLoss";
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
                                 Blob *dest_grad, bool overwrite) {
  MapMatrixType predict(dest_fea->MutableContent(), dest_fea->Height(),
                        dest_fea->Width());
  //! here src_grad is actually the input data/label from DataLayer
  MapMatrixType data(src_grad->MutableContent(), src_grad->Height(),
                     src_grad->Width());
  MapMatrixType fea_grad(dest_grad->MutableContent(), dest_grad->Height(),
                         dest_grad->Width());
  if (overwrite)
    fea_grad = (predict - data) / predict.rows();
  else
    fea_grad += (predict - data) / predict.rows();
}
}  // namespace lapis
