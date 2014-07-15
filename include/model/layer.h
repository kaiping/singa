// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-05 19:49

#ifndef INCLUDE_MODEL_LAYER_H_
#define INCLUDE_MODEL_LAYER_H_

#include <Eigen/Core>
#include <vector>
#include <string>
#include <map>
#include "proto/lapis.pb.h"
#include "model/edge.h"


namespace lapis {
/**
 * Base layer class.
 * Child layers should implement the ::Forward(), ::Backward() functions for
 * backpropagation method;
 * TODO(wangwei) For layers that support contrastive convergence,
 * ::ComputeLayer() should be implemented.
 * Currently there are 5 layers of type "Logistic", "Data", "Softmax",
 * "EuclidenLoss", "RELU" respectively
 */
class Layer {
 public:
  /**
   * Set layer properties, e.g., name and num_outputs(feature dimension for
   * normal layer or num of filters for convolutional layer). Layers should be
   * created after edges, then we can set the out going and incoming edges in
   * this function.
   * @param layer_proto user defined layer configuration, it has the names of
   * out going and incoming edges, based on which the corresponding edges are
   * added to the layer. see LayerProto
   * @param edge_map a map from edge name to the edge object.
   */
  virtual void Init(const LayerProto &layer_proto,
                    const map<string, Edge *> &edge_map);
  /**
   * allocate memory for storing activations/features/gradients, etc.
   * @param batchsize num of instances processed in one mini-batch
   * @param alg algorithm to compute gradients, i.e., Backpropagation,
   * TODO (wangwei), add support for Contrastive Divergence
   * @param sources data providers, can be null for layers do not accept inputs
   */
  virtual void Setup(int batchsize, Trainer::Algorithm alg,
                     const vector<DataSource *> &sources) = 0;
  /**
   * Forward propagate features through the Net
   * It aggregates activations from all incoming edges, and then apply the
   * activation function
   */
  virtual void Forward() = 0;
  /**
   * Backward propagate gradients through the Net
   * It aggregates gradients from all outgoing edges, and then computes
   * the gradient w.r.t the aggregated activation.
   */
  virtual void Backward() = 0;
  /**
   * Combine momentum, learning rate, weight decay, etc,  with the gradients
   * of parameters associated with this layer
   * @param trainer Trainer pointer which provides the hyper-parameters for
   * computing updates
   */
  virtual void ComputeParamUpdates(const Trainer *trainer) = 0;
  /**
   * Marshal layer properties and parameters into google protobuf object
   * @param proto see LayerProto in lapis.proto
   */
  virtual void ToProto(LayerProto *layer_proto);
  /**
   * Return true for layers accept input data, e.g., DataLayer,
   * false for all other layer
   */
  virtual inline bool HasInput() = 0;
  /**
   * Return the output feature Blob of this layer connected to the edge
   * @param edge which connects to the feature to be returned
   */
  virtual inline Blob &Feature(Edge *edge) = 0;
  /**
   * Return the gradient Blob connected to the edge.
   * Usually, it is the gradient of activations, which will be back propagated
   * to lower layers. But for DataLayer, it returns the feature Blob, because
   * the edge is an loss Edge, which computes the gradient by comparing
   * prediction and the data (e.g., label).
   * @param edge which connectes to the gradient
   */
  virtual inline Blob &Gradient(Edge *edge) = 0;
  /**
   * Add out going edge
   * @param edge out going edge associated with this layer
   */
  void AddOutEdge(const Edge *edge) {
    out_edges_.push_back(edge);
  }
  /**
   * Add incoming edge
   * @param edge incoming edge associated with this layer
   */
  void AddInEdge(const Edge *edge) {
    in_edges_.push_back(edge);
  }
  /**
   * Return name of this layer
   */
  const string &Name() {
    return name_;
  }

 protected:
  vector<Edge *> out_edges_;
  vector<Edge *> in_edges_;
  string name_;
};

}  // namespace lapis

#endif  // INCLUDE_MODEL_LAYER_H_
