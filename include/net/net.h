// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-14 13:12

#ifndef INCLUDE_MODEL_NET_H_
#define INCLUDE_MODEL_NET_H_

#include <google/protobuf/repeated_field.h>
#include <vector>
#include "net/layer.h"
#include "net/edge.h"
#include "net/param.h"
#include "datasource/data_source.h"
#include "proto/model.pb.h"


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
  explicit Net(const NetProto &net_proto);
  void Setup(const char flag,int batchsize,
             const std::map<std::string, Shape> &shapes,
             const std::map<std::string, int> & stores);
  void Setup(const char flag,int batchsize,
             const std::map<std::string, Shape> &shapes);

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
