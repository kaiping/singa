// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-05 19:49

#ifndef INCLUDE_MODEL_LAYER_H_
#define INCLUDE_MODEL_LAYER_H_

#include <vector>
#include <string>
#include <map>
#include <functional>
#include <memory>

#include "proto/model.pb.h"
#include "net/edge.h"
#include "net/trainer.h"
#include "net/param.h"
#include "datasource/data_source.h"

/**
 * \file this file includes the declarations of both Layer and LayerFactory
 */

namespace lapis {
/**
 * forward declaration of Trainer.
 */
class Trainer;
/**
 * forward declaration of Edge.
 */
class Edge;
/**
 * Base layer class.
 * Child layers should implement the ::Forward(), ::Backward() functions for
 * backpropagation method;
 * TODO(wangwei) For layers that support contrastive convergence,
 * ::ComputeLayer() should be implemented.
 * Currently there are 3 layers of type LogisticLayer, DataLayer and
 * LinearLayer. The identifier of each layer is defined as id field in the
 * corresponding class
 */
class Layer {
 public:
   virtual ~Layer(){}
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
  virtual void Init(const LayerProto &proto);
  /**
   * allocate memory for storing activations/features/gradients, etc.
   * @param batchsize num of instances processed in one mini-batch
   * @param alg algorithm to compute gradients, i.e., Backpropagation,
   * TODO (wangwei), add support for Contrastive Divergence
   * @param sources data providers, can be null for layers do not accept inputs
   */
  virtual void Setup(const char flag);
  /**
   * Forward propagate features through the Net
   * It aggregates activations from all incoming edges, and then apply the
   * activation function
   */
  virtual void Forward();
  /**
   * Backward propagate gradients through the Net
   * It aggregates gradients from all outgoing edges, and then computes
   * the gradient w.r.t the aggregated activation.
   */
  virtual void Backward();
  /**
   * Combine momentum, learning rate, weight decay, etc,  with the gradients
   * of parameters associated with this layer
   * @param trainer Trainer pointer which provides the hyper-parameters for
   * computing updates; May need cast it into SGDTrainer, to get momentum,
   * weight_decay, etc.
   */
  virtual void ComputeParamUpdates(const Trainer *trainer);
  /**
   * Apply dropout to src blob and write to dest blob, record the mask
   * @param drop_prob probability for drop out a neuron
   * @param scale a scale factor to be applied to every neuron after dropout
   * @param src the blob whoes data is going to be dropped
   * @param dest the blob to write the data after dropout
   * @param mask record which neuron is dropped for back propagating gradients,
   * if mask[i]=0, then the i-th neuron is dropped.
   */
  virtual void Dropout(float drop_prob, const Blob &src,
                       Blob *dest, Blob *mask);
  /**
   * Back propagate the gradient for dropout operation
   * @param scale a scale factor multiplied to the gradient
   * @param src gradient from top blob
   * @param dest blob to store the gradient after dropout operation
   * @param mask it records which neuron was dropped in ::Dropout(), and
   * directs how to pass the gradient from src to dest.
   */
  virtual void ComputeDropoutGradient(float drop_prob, const Blob &src ,
                                      const Blob &mask, Blob *dest);
  /**
   * Marshal layer properties and parameters into google protobuf object
   * @param proto see LayerProto in lapis.proto
   */
  virtual void ToProto(LayerProto *layer_proto);
  /**
   * Return true for layers accept input data, e.g., DataLayer,
   * false for all other layer; default is false;
   */
  virtual bool HasInput() {
    return false;
  }
  /**
   * Return the output feature Blob of this layer connected to the edge
   * @param edge which connects to the feature to be returned
   */
  virtual Blob &feature(Edge *edge)=0;
  /**
   * Return the gradient Blob connected to the edge.
   * Usually, it is the gradient of activations, which will be back propagated
   * to lower layers. But for DataLayer, it returns the feature Blob, because
   * the edge is an loss Edge, which computes the gradient by comparing
   * prediction and the data (e.g., label).
   * @param edge which connectes to the gradient
   */
  virtual Blob &gradient(Edge *edge)=0;
  /**
   * Return parameters of this layer

    std::vector<Param *> &params() {
      return params_;
    }
   */
  /**
   * Return name of this layer
   */
  const std::string &name() {
    return name_;
  }
  void add_in_edge(Edge *edge) {
    in_edges_.push_back(edge);
  }
  void add_out_edge(Edge *edge) {
    out_edges_.push_back(edge);
  }
  /**
   * Return outgoing edges
   */
  std::vector<Edge *> &out_edges() {
    return out_edges_;
  }
  /**
   * Return incoming edges
   */
  std::vector<Edge *> &in_edges() {
    return in_edges_;
  }

 protected:
  float drop_prob_;
  std::string name_, type_;
  std::vector<Edge *> out_edges_;
  std::vector<Edge *> in_edges_;
  Blob drop_fea_, drop_grad_, mask_;
};

/****************************************************************************/
/**
 * Register Layer with identifier ID
 * @param ID identifier of the layer e.g., Logistic, i.e., the type field
 * in LayerProto
 * @param LAYER the child layer class
 */
#define REGISTER_LAYER(ID, LAYER) LayerFactory::Instance()->\
  RegisterCreateFunction(ID, [](void)-> Layer* {return new LAYER();})

/**
 * Factory for creating layer based on user provided layer type/identifier.
 * Users are required to register user-defined layers before creating instances
 * of them during runtime. For example, if you define a new Layer FooLayer with
 * identifier "Foo", then you can use it in your net by 1) configure your
 * layer proto with the type field to be "Foo". 2) register it (e.g., at the
 * start of the program). Then your FooLayer will be created by calling
 * LayerFactory::Instance()->Create("Foo");
 */
class LayerFactory {
 public:
  /**
   * static method to get instance of this factory
   */
  static std::shared_ptr<LayerFactory> Instance();
  /**
   * Register user defined layer, i.e., add the layer type/identifier and a
   * function which creats an instance of the layer. This function is called by
   * the REGISTER_LAYER macro.
   * @param id identifier of the layer, every layer has a type to identify it
   * @param create_function a function that creates a layer instance
   */
  void RegisterCreateFunction(const std::string id,
                              std::function<Layer*(void)> create_function);
  /**
   * create a layer  instance by providing its type
   * @param type the identifier of the layer to be created
   */
  Layer *Create(const std::string id);

 private:
  //! To avoid creating multiple instances of this factory in the program
  LayerFactory();
  //! Map that stores the registered Layers
  std::map<std::string, std::function<Layer*(void)>> layer_map_;
  static std::shared_ptr<LayerFactory> instance_;
};

}  // namespace lapis

#endif  // INCLUDE_MODEL_LAYER_H_
