// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-05 19:49

#ifndef INCLUDE_NET_LAYER_H_
#define INCLUDE_NET_LAYER_H_

#include <vector>
#include <string>
#include <map>
#include <functional>
#include <utility>
#include <memory>

#include "proto/model.pb.h"
#include "net/edge.h"
#include "net/param.h"
#include "da/dary.h"
#include "utils/common.h"
#include "utils/timer.h"

/**
 * \file this file includes the declarations of both Layer and LayerFactory
 */

using std::vector;
using std::string;
namespace lapis {
/**
 * forward declaration of Edge.
 */
class Edge;
using StrStrEdge=std::map<std::pair<string, string>, Edge*>;
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
   Layer(){}
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
  virtual void Init(const LayerProto &proto, StrStrEdge *edge_map);
  virtual void InitDAryShape(const vector<vector<int>>& shapes);
  virtual void InitDAryShape();
  virtual void CollectParams(vector<Param*> *params);
  virtual vector<Param*> GetParams();

  /**
   * partition the layer along k-th dimension starting from 0
   * the parameters should be partitioned according to the
   * partition of the layer; Called after InitDAryShape;
   */
  void SetupDAry(int pdim);
  void SetPartition(int pdim);
  /**
   * Forward propagate features through the Net
   * It aggregates activations from all incoming edges, and then apply the
   * activation function
   */
    virtual void ComputeFeature();
  /**
   * Backward propagate gradients through the Net
   * It aggregates gradients from all outgoing edges, and then computes
   * the gradient w.r.t the aggregated activation.
   */
    virtual void ComputeGradient();
 /**
   * Marshal layer properties and parameters into google protobuf object
   * @param proto see LayerProto in lapis.proto
   */
  virtual void ToProto(LayerProto *layer_proto, bool copyData);
  /**
   * Return true for layers accept input data, e.g., DataLayer,
   * false for all other layer; default is false;
   */
  virtual bool HasInput() {
    return false;
  }
  virtual bool HasOutput() {
    return false;
  }

  /**
   * Return the output feature Blob of this layer connected to the edge
   * @param edge which connects to the feature to be returned
   */
  virtual const DAry &GetData(Edge *edge){
    return data_;
  }
  virtual DAry *GetMutableData(Edge *edge){
    return &data_;
  }

  /**
   * Return the gradient Blob connected to the edge.
   * Usually, it is the gradient of activations, which will be back propagated
   * to lower layers. But for DataLayer, it returns the feature Blob, because
   * the edge is an loss Edge, which computes the gradient by comparing
   * prediction and the data (e.g., label).
   * @param edge which connectes to the gradient
   */
  virtual DAry * GetMutableGrad(Edge *edge){
    return &grad_;
  }
  virtual const DAry& GetGrad(Edge *edge){
    return grad_;
  }
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
class ConvLayer: public Layer {
 public:
  virtual void Init(const LayerProto &proto, StrStrEdge *edge_map);
  virtual void InitDAryShape();
  virtual void SetupDAry(int pdim);
  virtual void SetPartition(int pdim);
  virtual void ComputeFeature();
  virtual void ComputeGradient();
  virtual void CollectParams(vector<Param*> *params);
  virtual vector<Param*> GetParams();
  virtual void ToProto(LayerProto *layer_proto, bool copyData);
  void Img2Col(DAry* dst, const DAry& src);
  void Col2Img(DAry* dst, const DAry& src);
 private:
  //! the feature (e.g., input image) shape for the bottom layer
  int num_, channels_, height_, width_;
  //! shape for conv image
  int cheight_, cwidth_;
  //! group weight height, width (col height), and col width
  int M_, K_, N_;
  //! num of groups, from caffe
  int ngroups_;
  //! height and width of the kernel/filter, assume the kernel is square
  int wsize_;
  //! length/width between to successive kernels/filters
  int stride_;
  //! padding size for boundary rows and cols
  int pad_;
  //! number of kernels
  int nkernels_;
  //! one row per kernel; shape is num_kernels_*(channels_*kernel_size^2)
  Param weight_ ;
  //! the length is nkernels_
  Param bias_;
  //! store result of image to column, TODO create ColLayer
  DAry col_data_, col_grad_;

  Timer t;
 public:
  float img2col, col2img, tdot, tadd;
};

class ReLULayer: public Layer {
 public:
  virtual void InitDAryShape();
  virtual void ComputeFeature();
  virtual void ComputeGradient();
};

class DropoutLayer: public Layer {
 public:
  virtual void Init(const LayerProto &proto, StrStrEdge *edge_map);
  virtual void InitDAryShape();
  virtual void SetPartition(int pdim);
  virtual void SetupDAry(int pdim);
  virtual void ComputeFeature();
  virtual void ComputeGradient();
  virtual void ToProto(LayerProto *layer_proto, bool copyData);
 private:
  float drop_prob_;
  /* record which neuron is dropped, required for back propagating gradients,
   * if mask[i]=0, then the i-th neuron is dropped.
   */
  DAry mask_;
};

class PoolingLayer: public Layer {
 public:
  virtual void Init(const LayerProto &proto, StrStrEdge *edge_map);
  virtual void InitDAryShape();
  virtual void ComputeFeature();
  virtual void ComputeGradient();
  virtual void ToProto(LayerProto *layer_proto, bool copyData);
 private:
  //! pooling window size and stride
  int wsize_, stride_;
  //! shape for bottom layer feature
  int channels_, height_, width_;
  //! shape after pooling
  int pheight_, pwidth_;
  //! batchsize
  int num_;
  LayerProto::PoolingMethod pooling_method_;
};

class LRNLayer: public Layer {
/**
 * Local Response Normalization edge
 * b_i=a_i/x_i^beta
 * x_i=knorm+alpha*\sum_{j=max(0,i-n/2}^{min(N,i+n/2}(a_j)^2
 * n is size of local response area.
 * a_i, the activation (after ReLU) of a neuron convolved with the i-th kernel.
 * b_i, the neuron after normalization, N is the total num of kernels
 */

 public:
  virtual void Init(const LayerProto &proto, StrStrEdge *edge_map);
  virtual void InitDAryShape();
  virtual void SetupDAry(int pdim);
  virtual void SetPartition(int pdim);
  virtual void ComputeFeature();
  virtual void ComputeGradient();
  virtual void ToProto(LayerProto *layer_proto, bool copyData);

 private:
  //! shape of the bottom layer feature
  int num_, channels_, height_, width_;
  //! size local response (neighbor) area and padding size
  int wsize_, lpad_, rpad_;
  //! hyper-parameter
  float alpha_, beta_, knorm_;
  DAry norm_, ratio_; //ratio : grad/(data*norm)
};
class FCLayer: public Layer {
  /*
   * fully connected layer
   */
 public:
  virtual void Init(const LayerProto &proto, StrStrEdge *edge_map);
  virtual void InitDAryShape();
  virtual void SetupDAry(int pdim);
  virtual void SetPartition(int pdim);
  virtual void ComputeFeature();
  virtual void ComputeGradient();
  virtual void CollectParams(vector<Param*> *params);
  virtual vector<Param*> GetParams();
  virtual void ToProto(LayerProto *layer_proto, bool copyData);
 private:
  //! dimension of the hidden layer
  int hdim_;
  //! dimension of the visible layer
  int vdim_;
  int num_;
  Param weight_, bias_;
};

class OutputLayer: public Layer{
 public:
  virtual Performance CalcPerf(bool loss, bool accuracy);
};
class SoftmaxLossLayer: public OutputLayer {
  /*
   * connected from the label layer and the last fc layer
   */
 public:
  virtual void Init(const LayerProto &proto, StrStrEdge *edge_map);
  virtual bool HasOutput() {return true;}
  virtual void InitDAryShape();
  virtual void SetupDAry(int pdim);
  virtual void SetPartition(int pdim);
  virtual void ComputeFeature();
  virtual void ComputeGradient();
  virtual void ToProto(LayerProto *layer_proto, bool copyData);
  virtual Performance CalcPerf(bool loss, bool accuracy);
 private:
  int num_;
  int dim_;
  int topk_;
};

class InputLayer: public Layer {
 public:
  virtual void Init(const LayerProto &proto, StrStrEdge *edge_map);
  virtual bool HasInput() { return true; }
  virtual void AddInputRecord(const Record& record)=0;
  virtual void SetInputData(DAry *data);
  virtual const DAry& GetGrad(Edge* edge) {
    LOG(ERROR)<<"input layer has no grad";
    return grad_;
  }
  virtual DAry* GetMutableGrad(Edge* edge) {
    return nullptr;
  }
 protected:
  //DAry prefetch_data_; use the grad_ field for prefetch data
  int offset_;
};
class ImageLayer: public InputLayer {
 public:
  virtual void Init(const LayerProto &proto, StrStrEdge *edge_map);
  virtual void ToProto(LayerProto *layer_proto, bool copyData);
  virtual void InitDAryShape(const vector<vector<int>>& shapes);
  virtual void AddInputRecord(const Record& record);

 private:
  bool mirror_;
  int cropsize_;
  //int batchsize_, channels_, height_, width_;
};

class LabelLayer: public InputLayer {
 public:
  virtual void InitDAryShape(const vector<vector<int>>& shapes);
  virtual void AddInputRecord(const Record& record);
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

#endif  // INCLUDE_NET_LAYER_H_
