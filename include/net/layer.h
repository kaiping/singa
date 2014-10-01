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
   */
  virtual void Setup(const char flag);
  /**
   * Forward propagate features through the Net
   * It aggregates activations from all incoming edges, and then apply the
   * activation function
    virtual void Forward();
   */
  /**
   * Backward propagate gradients through the Net
   * It aggregates gradients from all outgoing edges, and then computes
   * the gradient w.r.t the aggregated activation.
      virtual void Backward();
   */
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
  virtual DAry &data(Edge *edge)=0;
  /**
   * Return the gradient Blob connected to the edge.
   * Usually, it is the gradient of activations, which will be back propagated
   * to lower layers. But for DataLayer, it returns the feature Blob, because
   * the edge is an loss Edge, which computes the gradient by comparing
   * prediction and the data (e.g., label).
   * @param edge which connectes to the gradient
   */
  virtual DAry &grad(Edge *edge)=0;

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
  const std::vector<Edge *> &out_edges() {
    return out_edges_;
  }
  /**
   * Return incoming edges
   */
  const std::vector<Edge *> &in_edges() {
    return in_edges_;
  }

 protected:
  std::string name_, type_;
  std::vector<Edge *> out_edges_;
  std::vector<Edge *> in_edges_;
  DAry data_, grad_;
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

/**
 * Layer for fetching raw input features
 * It setups DataSource firstlyn ::Setup() and then fetch the data batch in
 * ::Forward().
 */
class DataLayer : public Layer {
 public:
  /**
   * Identifier of this layer, the value is "Data".
   */
  static const std::string kType;
  /**
   * Set data source identifier, i.e. name.
   */
  virtual void Init(const LayerProto &proto);
  /**
   * Set the input batch shape, including batchsize, channels, height, width.
   * @param shape
   */
  void SetInputShape(int batchsize, const Shape &data_shape);
  void SetInputStore(int store_id);
  void LoadData(const DAry &input, Phase phase);
  /**
   * allocate memory
   */
  virtual void Setup(const char flag);
  /**
   * fetch data from data source
   */
  virtual void Forward();
  /*
   * Just call Backward function of out going edges.
  virtual void Backward();
   */
  /**
   * Write the data source name
   */
  virtual void ToProto(LayerProto *layer_proto);
  virtual bool HasInput() {
    return true;
  }
  /**
   * @param edge if edge is nullptr it means this function is called to fill
   * new records for the layer. hence return the tmp blob if the image should
   * be croped; otherwise return the data blob
   */
  virtual const Blob &data(Edge *edge) {
      return data_;
  }
  /**
   * Because DataLayer is usually connected from loss edges, hence this
   * function returns the data provided by DataSource to the loss edge to
   * compute the gradients.
   * @param edge not used currently.
   */
  virtual const Blob &grad(Edge *edge) {
    return data_;
  }

  inline int store_id() {
    return  store_id_;
  }

  inline std::string& data_source() {
    return data_source_;
  }

 private:
  bool mirror_;
  int cropsize_;
  int batchsize_, channels_,height_,width_;
  std::string data_source_;
  int store_id_;
};


}  // namespace lapis

#endif  // INCLUDE_MODEL_LAYER_H_
