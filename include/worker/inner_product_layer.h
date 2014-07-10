// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-08 15:24

#ifndef INCLUDE_WORKER_INNER_PRODUCT_LAYER_H_
#define INCLUDE_WORKER_INNER_PRODUCT_LAYER_H_

#include <Eigen/Core>
#include <map>
#include <string>
#include "proto/lapis.pb.h"

namespace lapis {
class InnerProductLayer : public Layer {
 public:
  virtual void setup(const LayerProto& layer_proto,
                     std::map<string, Edge*>* edges);
  virtual void forward();
  virtual void backward();
  ~InnerProductLayer();
 private:
  MapMatrixType weight_(nullptr), weight_grad_(nullptr);
                out_(nullptr), out_grad_(nullptr),
                in_(nullptr), in_grad_(nullptr);
  MapVectorType bias_(nullptr), bias_grad_(nullptr);
};
}  // namespace lapis
#endif  // INCLUDE_WORKER_INNER_PRODUCT_LAYER_H_
