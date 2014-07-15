// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-14 13:12

#ifndef INCLUDE_MODEL_NET_H_
#define INCLUDE_MODEL_NET_H_

#include <vector>
namespace lapis {
/**
 * The neural network consists of Layers and Edges.
 */
class Net {
 public:
  void Init(const NetProto &net);
  inline vector<Layer *> &Layers() {
    return layers_;
  }
  inline vector<Edge *> &Edges() {
    return edges_;
  }
 private:
  vector<Layer *> layers_;
  vector<Edge *> edges_;
}

}  // namespace lapis
#endif  // INCLUDE_MODEL_NET_H_
