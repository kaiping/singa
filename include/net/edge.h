// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-10 22:29

#ifndef INCLUDE_NET_EDGE_H_
#define INCLUDE_NET_EDGE_H_
#include <string>
#include "net/layer.h"
#include "darray/darray.h"

using std::string;

namespace lapis {
class Layer;
/***************************************************************************
 * Edge Classes
 **************************************************************************/
/**
 * Base edge class.
 * One edge connects two layers (nodes). The edge can be directed, e.g., in feed
 * forward neural network, or undirected, e.g., in RBM and DBM. In DBN,
 * there are both directed edges and undirected edges.
 * We use edges just to express the connections between layers and pass data
 * between layers;
 */
class Edge {
 public:
   virtual ~Edge(){}
   Edge():node1_(nullptr), node2_(nullptr){}
  /**
   * return the other side of this edge w.r.t, layer
   * @param layer one side of the edge
   */
  Layer *OtherSide(const Layer *layer) {
    return node1_ == layer ? node2_ : node1_;
  }
  const std::string &GetName() {
    return node1_->name()+"-"+node2_->name();
  }
  const DArray& GetData(Layer* tolayer);

  DArray* GetMutableData(Layer* tolayer);

  const DArray& GetGrad(Layer* tolayer);

  DArray* GetMutableGrad(Layer* tolayer);

  /**
   * used for directed edge
   */
  Layer* src() {
    return node1_;
  }
  Layer* dst() {
    return node2_;
  }
  void set_src(Layer* src) {
    node1_=src;
  }
  void set_dst(Layer* dst) {
    node2_=dst;
  }
  /*
  DArray* GetPos(Layer* tolayer);
  DArray* GetNeg(Layer* tolayer);
  */
 private:
  /**
   * Sides/endpoints of the edge.
   * Normally for feed forward neural network, the edge direction is from
   * node1 to node2.
   */
  Layer *node1_, *node2_;
  bool is_directed_;
};
}  // namespace lapis
#endif  // INCLUDE_NET_EDGE_H_
