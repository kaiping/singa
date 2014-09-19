// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-24 14:22
#ifndef INCLUDE_MODEL_SOFTMAX_LOSS_EDGE_H_
#define INCLUDE_MODEL_SOFTMAX_LOSS_EDGE_H_

#include "net/edge.h"
#include "proto/model.pb.h"
#include "net/lapis.h"


namespace lapis {
class SoftmaxLossEdge : public Edge {
 public:
  virtual void Setup(const char flag);
  virtual void Forward(const Blob &src, Blob *dest, bool overwrite);
  virtual void Backward(const Blob &src_fea, const Blob &src_grad,
                        const Blob &dest_fea, Blob *dest_grad,
                        bool overwirte);

  void SetupTopBlob(bool alloc, Blob* blob);
  virtual char PrefixAbbrev() {return 's';}
 private:
  //! batch size
  int num_;
  //! dimension of the feature of this layer
  int dim_;
};
}  // namespace lapis
#endif  // INCLUDE_MODEL_SOFTMAX_LOSS_EDGE_H_
