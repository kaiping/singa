// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-15 19:42

#ifndef INCLUDE_MODEL_EUCLIDEAN_LOSS_EDGE_H_
#define INCLUDE_MODEL_EUCLIDEAN_LOSS_EDGE_H_
#include <string>
#include "model/blob.h"
#include "model/edge.h"
namespace lapis {
class EuclideanLossEdge : public Edge {
 public:
  static const std::string type;
  virtual void ToProto(EdgeProto *proto);
  virtual void Forward(const Blob *src_fea, Blob *dest_fea, bool overwrite);
  virtual void Backward(const Blob *src_fea, const Blob *src_grad,
                        const Blob *dest_fea, Blob *dest_grad, bool overwrite);
};

}  // namespace lapis
#endif  // INCLUDE_MODEL_EUCLIDEAN_LOSS_EDGE_H_
