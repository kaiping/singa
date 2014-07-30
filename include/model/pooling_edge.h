// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-23 16:56
#ifndef INCLUDE_MODEL_POOLING_EDGE_H_
#define INCLUDE_MODEL_POOLING_EDGE_H_

#include "proto/model.pb.h"
#include "model/edge.h"

namespace lapis {
/**
 * Pooling is to summary local response area using one value.
 * max pooling select the max one among the local area.
 * avg pooling average the local area.
 */
class PoolingEdge : public Edge {
 public:
  virtual void Init(const EdgeProto &proto,
                 const std::map<std::string, Layer *> &layer_map);
  virtual void ToProto(EdgeProto *proto);
  virtual void Forward(const Blob4 *src, Blob4 *dest, bool overwrite);
  virtual void Backward(const Blob4 *src_fea, const Blob4 *src_grad,
                        const Blob4 *dest_fea, Blob4 *dest_grad,
                        bool overwrite);

  virtual void SetupTopBlob(Blob4* blob);
 private:
  //! pooling kernel shape
  int kernel_size_, stride_;
  //! shape for bottom layer feature
  int channels_, height_,width_;
  //! shape after pooling
  int pool_height_, pool_width_;
  //! batchsize
  int num_;
  EdgeProto::PoolingMethod pooling_method_;
};
}  // namespace
#endif  // INCLUDE_MODEL_POOLING_EDGE_H_
