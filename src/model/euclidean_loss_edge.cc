// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-15 20:15

#include <glog/logging.h>

#include "utils/lapis.h"
#include "model/euclidean_loss_edge.h"

namespace lapis {

const std::string type = "EuclideanLoss";

void EuclideanLossEdge::ToProto(EdgeProto *proto) {
  Edge::ToProto(proto);
}

void EuclideanLossEdge::Forward(const Blob *src_fea, Blob *dest_fea,
                                bool overwrite) {
  LOG(INFO) << "Forward() of EuclideanLossEdge should not be called\n";
}
void EuclideanLossEdge::Backward(const Blob *src_fea, const Blob *src_grad,
                                 const Blob *dest_fea, Blob *dest_grad,
                                 bool overwrite) {
  MMat predict(dest_fea->mutable_data(), dest_fea->height(),
                        dest_fea->width());
  //! here src_grad is actually the input data/label from DataLayer
  MMat label(src_grad->mutable_data(), src_grad->height(),
                     src_grad->width());
  MMat fea_grad(dest_grad->mutable_data(), dest_grad->height(),
                         dest_grad->width());
  if (overwrite)
    fea_grad = (predict - label) / predict.rows();
  else
    fea_grad += (predict - label) / predict.rows();
}
}  // namespace lapis
