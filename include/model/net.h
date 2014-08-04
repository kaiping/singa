// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-14 13:12

#ifndef INCLUDE_MODEL_NET_H_
#define INCLUDE_MODEL_NET_H_

#include <vector>
#include "model/layer.h"
#include "model/edge.h"
#include "model/param.h"
namespace lapis {
/**
 * Forward declaration of Edge and Layer
 */
class Edge;
class Layer;
/**
 * The neural network consists of Layers and Edges.
 */
class Net {
 public:
  void Init(const NetProto &net);
  void ToProto(NetProto *net_proto);
  std::vector<Layer *> &layers() {
    return layers_;
  }
  std::vector<Edge *> &edges() {
    return edges_;
  }
  const std::vector<Param *> &params() {
    return params_;
  }
  ~Net();
 private:
  std::vector<Layer *> layers_;
  std::vector<Edge *> edges_;
  std::vector<Param *> params_;
};

}  // namespace lapis
#endif  // INCLUDE_MODEL_NET_H_
