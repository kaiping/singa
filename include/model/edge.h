// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-10 22:29

#ifndef INCLUDE_MODEL_EDGE_H_
#define INCLUDE_MODEL_EDGE_H_
#include <vector>
#include <string>

namespace lapis {
/**
 * Base edge class.
 * One edge connects two layers. The edge can be directed, e.g., in feed
 * forward neural network, or undirected, e.g., in RBM and DBM. In DBN,
 * there are both directed edges and undirected edges.
 * Normally, the edge contains parameters. It operates on output features
 * (or gradient of activations) of one layer and assign the results to
 * activations (or gradient of output features) of another layer.
 * Currently, 2 types of edges are supported, i.e., inner_product_edge and
 * eucldiean_loss_edge. TODO(wangwei)  implement convolution_edge etc.
 */
class Edge {
 public:
  /**
   * Set edge properties, allocate memory for parameters and initialize them
   * @param edge_proto user defined edge properties, e.g., edge name,
   * parameters, type
   */
  virtual void Init(const EdgeProto &edge_proto) = 0;
  /**
   * Marshal edge properties into google protobuf object
   */
  virtual void ToProto(EdgeProto *edge_proto) = 0;
  /**
   * Forward propagate feature, read from src and write to dest
   * @param src source feature
   * @param dest destination feature/activation to be set
   * #param overwrite if true overwrite the dest otherwise add it
   */
  virtual void Forward(const Blob *src, Blob *dest, bool overwrite) = 0;
  /**
   * Backward propagate gradient, read gradient from src and write to dest
   * @param src_grad read gradient/feature from src
   * @param dest_fea feature from the src layer
   * @param dest_grad write the comptued gradient to dest_grad
   * @param overwrite if true overwrite dest_grad otherwise add it
   */
  virtual void Backward(const Blob *src_grad, const Blob *dest_fea,
                        Blob *dest_grad) = 0;
  /**
   * return the other side of this edge w.r.t, layer
   * @param layer one side of the edge
   */
  inline Layer *OtherSide(const Layer *layer);
  /**
   * Set top end of this edge
   */
  inline void SetTop(const Layer *top);
  /**
   * Set bottom end of this edge
   */
  inline void SetBottom(const Layer *bottom);
  inline const Layer *Top();
  inline const Layer *Bottom();
  inline const string Name();

 protected:
  /**
   * Sides/endpoints of the edge.
   * Normally for feed forward neural network, the edge direction is from
   * bottom to top. The 'top' and 'bottom' just describe the positions of the
   * layers in the Net, hence it is possible that the direction of one edge is
   * from top to bottom, e.g., the EuclideanLossEdge is usually from the
   * highest (top) layer to an input layer (bottom), as in AutoEncoder.
   */
  Layer *top_, * bottom_;
  string name_;
};

}  // namespace lapis
#endif  // INCLUDE_MODEL_EDGE_H_
