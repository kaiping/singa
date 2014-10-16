// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-14 13:12

#ifndef INCLUDE_NET_NET_H_
#define INCLUDE_NET_NET_H_

#include <glog/logging.h>
#include <vector>
#include <map>
#include <stack>
#include "net/param.h"
#include "proto/model.pb.h"
#include "net/net.h"
#include "net/layer.h"
#include "net/edge.h"



namespace lapis {
/**
 * Forward declaration of Edge and Layer
 */
/**
 * The neural network consists of Layers and Edges.
 */
class Net {
 public:
  /**
   * construct the net structure, i.e., how edges and layers are connected
   */
  explicit Net(const NetProto &net_proto);
  void TimeOneBatch(int runs=50);
  void Forward();
  void Backward();
  void topology_sort(std::vector<Layer *> *layers) ;
  void topology_sort_inner(Layer *layer,
                         const std::map<Layer *,
                         std::vector<Layer *>> &adjacent_list,
                         std::map<Layer *, bool> *visited,
                         std::stack<Layer *> *stack) ;

  /**
   * setup the net by init dary shapes,
   * then allocate memory(optional) and init parameters (optional)
   */
  void InitDArrayShape();
  void InitDArrayShape(const vector<vector<int>>& shapes);
  void Setup() ;
  void Setup(const vector<vector<int>>& input_shapes) ;
  /**
   * set shapes of DArrays
   */
  void SetShape(const int batchsize, const Record &record);
  /**
   * allocate memory for DArrays
   */
  void AllocateMemory();
  /**
   * init parameters, must be called after InitDArrayShape
   * if memory of parameters are not allocated, do memory allocation before
   * init parameters
   */
  void InitParameters();
  void ToProto(NetProto *net_proto, bool copyData=false);
  const std::vector<InputLayer *> &input_layer() {
    return input_layer_;
  }
  InputLayer * input_layer(int k) {
    CHECK_LT(k, input_layer_.size());
    return input_layer_[k];
  }
  OutputLayer* output_layer(int k) {
    CHECK_LT(k, output_layer_.size());
    return output_layer_[k];
  }
  const std::vector<Layer *>& layers() {
    return layers_;
  }
  const std::vector<Param *> &params() {
    return params_;
  }
  ~Net();
 private:
  std::vector<Layer *> layers_;
  std::vector<OutputLayer *> output_layer_;
  std::vector<InputLayer *> input_layer_;
  std::vector<Edge *> edges_;
  std::vector<Param *> params_;
};

}  // namespace lapis
#endif  // INCLUDE_NET_NET_H_
