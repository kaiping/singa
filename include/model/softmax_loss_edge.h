// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-24 14:22
#ifndef INCLUDE_MODEL_SOFTMAX_LOSS_EDGE_H_
#define INCLUDE_MODEL_SOFTMAX_LOSS_EDGE_H_

#include "model/edge.h"
#include "proto/model.pb.h"

namespace lapis {
class SoftmaxLossEdge : public Edge {
 public:
  virtual void Setup(bool set_param);
  virtual void Forward(const Blob4 &src, Blob4 *dest, bool overwrite);
  virtual void Backward(const Blob4 &src_fea, const Blob4 &src_grad,
                        const Blob4 &dest_fea, Blob4 *dest_grad,
                        bool overwirte);

 private:
  //! batch size
  int num_;
  //! dimension of the feature of this layer
  int dim_;
  //! prob after softmax norm
  Blob2 prob_;
};
}  // namespace lapis
#endif  // INCLUDE_MODEL_SOFTMAX_LOSS_EDGE_H_
