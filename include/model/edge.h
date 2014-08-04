// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-10 22:29

#ifndef INCLUDE_MODEL_EDGE_H_
#define INCLUDE_MODEL_EDGE_H_
#include <string>
#include <map>
#include <vector>

#include "model/trainer.h"
#include "model/lapis.h"
#include "model/layer.h"
#include "model/param.h"
#include "proto/model.pb.h"

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
 * (or gradient of activations) of one layer and assigns the results to
 * activations (or gradient of output features) of another layer.
 */
class Edge {
 public:
   virtual ~Edge(){}
  /**
   * Set edge properties,
   * @param edge_proto user defined edge properties, e.g., edge name,
   * parameters, type
   */
  virtual void Init(const EdgeProto &proto,
                    const std::map<std::string, Layer *> &layer_map);
  /**
   * Setup properties of this edge based on bottom layer, e.g, parameter shape.
   * Allocate memory for parameters and initialize them according to user
   * specified init method. Some parameters can not be set until the bottom
   * layer is setup ready. May also allocate memory to store intermediate
   * results.
   * @param set_param set parameters; true for network init; false otherwise.
   */
  virtual void Setup(bool set_param);
  /**
   * Marshal edge properties into google protobuf object
   */
  virtual void ToProto(EdgeProto *proto);
  /**
   * Forward propagate feature, read from src and write to dest
   * @param src source feature
   * @param dest destination feature/activation to be set
   * #param overwrite if true overwrite the dest otherwise add it
   */
  virtual void Forward(const Blob &src, Blob *dest, bool overwrite)=0;
  /**
   * Backward propagate gradient, read gradient/feature blob from src and
   * feature blob from src, then compute the gradient for parameters of this
   * edge and dest layer.
   * @param src_fea feature (or activation) blob from the source layer that
   * connected to this edge
   * @param src_grad gradient blob from the source layer connected to this edge
   * @param dest_fea feature blob from the dest layer connected to this layer
   * @param dest_grad gradient blob from the dest layer connected to this edge,
   * If no need to compute that gradient, then set dest_grad=nullptr, e.g., if
   * the bottom layer is DataLayer, the no need to compute for the dest_grad.
   * @param overwrite if true overwrite dest_grad otherwise add to it
   */
  virtual void Backward(const Blob &src_fea, const Blob &src_grad,
                        const Blob &dest_fea, Blob *dest_grad,
                        bool overwrite)=0;
  /**
   * Combine hyper-paramters, e.g., momentum, learning rate, to compute
   * gradients of parameters associated with this edge, which will be
   * used to update the parameters. If there is no parameters, then do nothing.
   * Currently implemented as :
   * history=momentum*history-learning_rate*(gradient+weight_decay*param_data)
   * where history is the history update, param_data is the content of the
   * parameter, momentum, learning_rate, weight_decay are product of local
   * and global (i.e., from sgd trainer);
   * @param trainer contains hyper-parameters. May cast it into specific
   * trainer, e.g., SGDTrainer, to get momentum and weight_decay, etc.
   */
  virtual void ComputeParamUpdates(const Trainer *trainer);
  /**
   * Setup (Reshape) the blob from top layer connected to this edge. Because
   * the top blob is generated (although owned by the top layer) by this edge,
   * this edge will decide the shape of the blob and is responsible to setup it
   * @param blob the top blob to set setup.
   */
  virtual void SetupTopBlob(Blob *blob);
  /**
   * Return parameters associated this edge
   */
  std::vector<Param *> &params() {
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
  void set_top(Layer *top) {
    top_ = top;
  }
  /**
   * Set bottom end of this edge
   */
  void set_bottom(Layer *bottom) {
    bottom_ = bottom;
  }
  Layer *top() {
    return top_;
  }
  Layer *bottom() {
    return bottom_;
  }
  const std::string &name() {
    return name_;
  }

 protected:
  std::string name_,type_;
  /**
   * Sides/endpoints of the edge.
   * Normally for feed forward neural network, the edge direction is from
   * bottom to top. But for undirected edge, then top and bottom contains no
   * position information.
   */
  Layer *top_, * bottom_;
  std::vector<Param *> params_;
};


/****************************************************************************/
/**
 * Register Edge with type TYPE
 * @param TYPE identifier of the edge e.g., InnerProduct, i.e., the type field
 * in EdgeProto
 * @param EDGE the child edge class
 */
#define REGISTER_EDGE(TYPE, EDGE) EdgeFactory::Instance()->\
  RegisterCreateFunction(TYPE, [](void)-> Edge* {return new EDGE();});

/**
 * Factory for creating edge based on user provided edge type.
 * Users are required to register user-defined edges before creating instances
 * of them during runtime. For example, if you define a new Edge FooEdge with
 * type "Foo", then you can use it in your net by 1) configure your
 * edge proto with the type field to be "Foo". 2) register it (e.g., at the
 * start of the program). Then your FooEdge will be created by calling
 * EdgeFactory::Instance()->Create("Foo");
 */
class EdgeFactory {
 public:
  /**
   * static method to get instance of this factory
   */
  static std::shared_ptr<EdgeFactory> Instance();
  /**
   * Register user defined edge, i.e., add the edge type and a
   * function which creats an instance of the edge. This function is called by
   * the REGISTER_EDGE macro.
   * @param type type of the edge, every edge has a type to identify it
   * @param create_function a function that creates a edge instance
   */
  void RegisterCreateFunction(const std::string type,
                              std::function<Edge*(void)> create_function);
  /**
   * create a layer  instance by providing its type
   * @param type the type of the layer to be created
   */
  Edge *Create(const std::string type);

 private:
  //! To avoid creating multiple instances of this factory in the program
  EdgeFactory();
  //! Map that stores the registered Layers
  std::map<std::string, std::function<Edge*(void)>> edge_map_;
  static std::shared_ptr<EdgeFactory> instance_;
};

}  // namespace lapis
#endif  // INCLUDE_MODEL_EDGE_H_
