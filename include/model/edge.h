// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-10 22:29

#ifndef INCLUDE_MODEL_EDGE_H_
#define INCLUDE_MODEL_EDGE_H_
#include <string>
#include "model/trainer.h"
#include "model/blob.h"
#include "model/layer.h"
#include "model/param.h"
#include "proto/lapis.pb.h"

namespace lapis {
//! forward declaration for Layer.
class Layer;
//! forward declaration for Trainer.
class Trainer;
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
   * @param dest_grad write the comptued gradient to dest_grad, if no need to
   * compute that gradient, then set dest_grad=nullptr
   * @param overwrite if true overwrite dest_grad otherwise add it
   */
  virtual void Backward(const Blob *src_grad, const Blob *dest_fea,
                        Blob *dest_grad, bool overwrite) = 0;

  /**
   * Combine hyper-paramters, e.g., momentum, learning rate, to compute
   * gradients of parameters associated with this edge, which will be
   * used to update the parameters. If there is no parameters, then do nothing.
   * @param trainer contains hyper-parameters. May cast it into specific
   * trainer, e.g., SGDTrainer, to get momentum and weight_decay, etc.
   */
  virtual void ComputeParamUpdates(const Trainer *trainer);
  /**
   * Return parameters associated this edge
   */
  std::vector<Param *> &Params() {
    return params_;
  }
  /**
   * return the other side of this edge w.r.t, layer
   * @param layer one side of the edge
   */
  Layer *OtherSide(const Layer *layer) {
    return top_ == layer ? bottom_ : top_;
  }
  /**
   * Set top end of this edge
   */
  void SetTop(Layer *top) {
    top_ = top;
  }
  /**
   * Set bottom end of this edge
   */
  void SetBottom(Layer *bottom) {
    bottom_ = bottom;
  }
  const Layer *Top() {
    return top_;
  }
  const Layer *Bottom() {
    return bottom_;
  }
  const std::string &Name() {
    return name_;
  }

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
  std::vector<Param *> params_ ;
  std::string name_;
};


/****************************************************************************/
/**
 * Register Edge with identifier ID
 * @param ID identifier of the edge e.g., InnerProduct, i.e., the type field
 * in EdgeProto
 * @param EDGE the child edge class
 */
#define REGISTER_EDGE(ID, EDGE) EdgeFactory::Instance()->\
  RegisterCreateFunction(ID,[](void)-> Edge* {return new EDGE();});

/**
 * Factory for creating edge based on user provided edge type/identifier.
 * Users are required to register user-defined edges before creating instances
 * of them during runtime. For example, if you define a new Edge FooEdge with
 * identifier "Foo", then you can use it in your net by 1) configure your
 * edge proto with the type field to be "Foo". 2) register it (e.g., at the
 * start of the program). Then your FooEdge will be created by calling
 * EdgeFactory::Instance()->Create("Foo");
 */
class EdgeFactory {
 public:
  /**
   * static method to get instance of this factory
   */
  static EdgeFactory *Instance();
  /**
   * Register user defined edge, i.e., add the edge type/identifier and a
   * function which creats an instance of the edge. This function is called by
   * the REGISTER_EDGE macro.
   * @param id identifier of the edge, every edge has a type to identify it
   * @param create_function a function that creates a edge instance
   */
  void RegisterCreateFunction(const std::string id,
                              std::function<Edge*(void)> create_function);
  /**
   * create a layer  instance by providing its type
   * @param type the identifier of the layer to be created
   */
  Edge *Create(const std::string id);

 private:
  //! To avoid creating multiple instances of this factory in the program
  EdgeFactory() {}
  //! Map that stores the registered Layers
  std::map<std::string, std::function<Edge*(void)>> layer_map_;
};

}  // namespace lapis
#endif  // INCLUDE_MODEL_EDGE_H_
