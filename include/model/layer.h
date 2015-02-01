#ifndef INCLUDE_NET_LAYER_H_
#define INCLUDE_NET_LAYER_H_

#include <vector>
#include <string>
#include <map>
#include <functional>
#include <utility>
#include <memory>
#include <chrono>
#include <random>

#include "proto/model.pb.h"
//#include "model/param.h"
#include "utils/common.h"

/**
 * \file this file includes the declarations of Layer and its children classes.
 */

using std::vector;
using std::string;
using std::map;
using std::shared_ptr;
using std::pair;

namespace singa {

/**
 * Base layer class.
 * Children should implement at least Layer::Setup, Layer::ComputeFeature(),
 * Layer::ComputGradient() functions for backpropagation method;
 * TODO(wangwei) implement children layers to support contrastive divergence,
 * The identifier of each layer is the literal string of the class name without
 * the suffix "Layer", which is used in net configuration and registration.
 */
class Layer {
 public:
  Layer(){}
  /**
   * construct layer from the other Layer, but with different shape
   */
  void Init(const Layer& other, const vector<int>& shape);
  virtual ~Layer(){}
  /**
   * initialize members, called after layer specific Init().
   * simply copy the configuations , most
   * initializations are done by Setup().
   * @param layer_proto user defined layer configuration
   */
  virtual void Init(const LayerProto &proto);
  /**
   * Marshal layer properties and data into google protobuf object
   * (i.e., snapshot).
   * Parameters are marshalled separately into another object (i.e., model).
   * @param layer_proto
   * @param copyData if true marshal data of DArray
   */
  virtual void ToProto(LayerProto *layer_proto, bool copyData);
  /**
   * Setup the shapes for data and parameters, also setup some properties.
   * @param src_layers layers connecting to this layer
   */
  virtual void Setup(const vector<shared_ptr<Layer>>& src_layers);
  /**
   * \copybrief Setup(const vector<shared_ptr<Layer>>& src_layers)
   */
  virtual void Setup();
  /**
   * this should not change the current shapes_. Check the shapes outside the
   * function.
   */
  virtual void SetupAfterPartition(const vector<shared_ptr<Layer>>& src_layers);
  /**
   * \copybrief SetupAfterPartition(const vector<shared_ptr<Layer>>& src_layers).
   */
  virtual void SetupAfterPartition();
  /**
   * collect parameters associated with this layer.
   * Layers that have paramters must overload this function.
   * parameter id is set in sequence order starting with 0.
   * @param params parameters collected from previous layers.
  virtual void CollectParams(vector<Param*> *params){};
   */
  /**
   * Layers that have paramters must overload this function.
   * @return parameters associated with this layer
  virtual vector<Param*> GetParams(){ return vector<Param*>(); }
   */
  /**
   * Compute features of this layer based on connected layers.
   * Implement forward propagation for BP; TODO Implement both postive phase
   * and negative phase for CD.
   * @param src_layers layers connecting to this layer
   */
  virtual void ComputeFeature(const vector<shared_ptr<Layer>>& src_layers)=0;
  /**
   * \copybrief ComputeFeature(const vector<shared_ptr<Layer>>& src_layers)
   */
  virtual void ComputeFeature();
  /**
   * Compute gradients for parameters and connecting layers.
   * Implement backward propagation for BP; TODO Calculate gradients for
   * parameters for CD.
   * @param src_layers layers connecting to this layer.
   */
  virtual void ComputeGradient(const vector<shared_ptr<Layer>>& src_layers)=0;
  /**
   * \copybrief ComputeGradient(const vector<shared_ptr<Layer>>& src_layers)
   */
  virtual void ComputeGradient();
  /**
   * decide on which dimension of DArray to do the partitioning.
   * @mode kModel, kData, kHybrid, kNone (no partition)
   * @return the partition dimension, -1 for no partition
   */
  virtual int partition_dimension() const {
    if(layer_proto_.partition_type()==kDataPartition)
      return 0;
    else if(layer_proto_.partition_type()==kLayerPartition)
      return 1;
    else if(layer_proto_.partition_type()==kNone)
      return -1;
  }

  virtual ConnectionType connection_type(int k) const {
    CHECK_LT(k, srclayers_.size());
    return kOneToOne;
  }
  virtual PartitionType partition_type() const {
    return layer_proto_.partition_type();
  }
  virtual void set_locationid(int id){
    layer_proto_.set_locationid(id);
  }
  virtual int locationid() const {
    return layer_proto_.locationid();
  }
  virtual void set_partitionid(int id){
    layer_proto_.set_partitionid(id);
  }
  virtual int partitiionID() const {
    return layer_proto_.partitionid();
  }
  virtual void set_name(string name){
    name_=name;
    layer_proto_.set_name(name);
  }
  virtual const string type() const {
    return layer_proto_.type();
  }
  /**
   * Return name of this layer
   */
  const std::string &name() const {
    return layer_proto_.name();
  }
  virtual const vector<int>& shapes(const Layer* layer) const{
    CHECK_LE(shapes_.size(),1);
    if(shapes_.size()==1)
      return shapes_[0];
    else{
      return vector<int>{};
    }
  }
  /**
   * @return a const ref for DArray storing neuron values of this layer for BP
  virtual const DArray& data() {return data_;}
   */
  /**
   * @return a const ref for DArray storing neuron grads of this layer for BP
  virtual const DArray& grad() {return grad_;}
  virtual DArray* mutable_data() {return &data_;}
  virtual DArray* mutable_grad() {return &grad_;}
   */

  virtual const shared_ptr<Layer> dstlayers(string name) const {
    if(dstlayers_.find(name)==dstlayers_.end())
      return nullptr;
    return dstlayers_.at(name);
  }
  virtual const shared_ptr<Layer> srclayers(string name) const {
    if(srclayers_.find(name)==srclayers_.end())
      return nullptr;
    return srclayers_.at(name);
  }
  virtual const map<string, shared_ptr<Layer>> srclayers() const {
    return srclayers_;
  }
  virtual const map<string, shared_ptr<Layer>> dstlayers() const {
    return dstlayers_;
  }
  virtual const int srclayers_size() const {
    return srclayers_.size();
  }
  virtual const int dstlayers_size() const {
    return dstlayers_.size();
  }

  virtual void remove_dstlayers(const shared_ptr<Layer>& layer){
    remove_dstlayers(layer->name());
    layer->remove_srclayers(name_);
  }
  virtual void remove_srclayers(const shared_ptr<Layer>& layer){
    remove_srclayers(layer->name());
    layer->remove_dstlayers(name_);
  }
  virtual void clear_srclayers() {
    for(auto& entry: srclayers_)
      remove_srclayers(entry.second);
  }
  virtual void clear_dstlayers() {
    for(auto& entry: dstlayers_)
      remove_dstlayers(entry.second);
  }
  virtual void add_srclayers(shared_ptr<Layer> myself, shared_ptr<Layer> layer){
    CHECK_EQ(myself.get(),this);
    add_srclayers(layer);
    layer->add_dstlayers(myself);
  }
  virtual void add_dstlayers(shared_ptr<Layer> myself,shared_ptr<Layer> layer){
    CHECK_EQ(myself.get(),this);
    add_dstlayers(layer);
    layer->add_srclayers(myself);
  }
  virtual void set_srclayers(shared_ptr<Layer> myself,
      vector<shared_ptr<Layer>>& layers){
    clear_srclayers();
    for(shared_ptr<Layer> layer: layers){
      add_srclayers(myself, layer);
    }
  }
  virtual void set_dstlayers(shared_ptr<Layer> myself,
      vector<shared_ptr<Layer>>& layers){
    clear_dstlayers();
    for(shared_ptr<Layer> layer: layers){
      add_dstlayers(myself, layer);
    }
  }

  shared_ptr<Layer> ordered_srclayers(int k){
    CHECK_LT(k, ordered_srclayers_.size());
    if(ordered_dstlayers_.size()!=dstlayers_.size())
      OrderConnectedLayers();
    return ordered_srclayers_[k];
  }
  shared_ptr<Layer> ordered_dstlayers(int k){
    CHECK_LT(k, ordered_dstlayers_.size());
    if(ordered_srclayers_.size()!=srclayers_.size())
      OrderConnectedLayers();
    return ordered_dstlayers_[k];
  }

  void OrderConnectedLayers(){
    vector<pair<int, string>> vec;
    ordered_srclayers_.clear();
    for(auto& entry: srclayer_order_)
      vec.push_back(std::make_pair(entry.second, entry.first));
    std::sort(vec.begin(),vec.end());
    for(auto& entry: vec)
      ordered_srclayers_.push_back(srclayers_[entry.second]);
    vec.clear();

    for(auto& entry: dstlayer_order_)
      vec.push_back(std::make_pair(entry.second, entry.first));
    std::sort(vec.begin(),vec.end());
    ordered_dstlayers_.clear();
    for(auto& entry: vec)
      ordered_dstlayers_.push_back(dstlayers_[entry.second]);
  }
 protected:
  void remove_srclayers(string name){
    srclayers_.erase(name);
    srclayer_order_.erase(name);
    auto iter=ordered_srclayers_.begin();
    while(iter!=ordered_srclayers_.end()&&(*iter)->name()!=name)
      iter++;
    CHECK_EQ((*iter)->name(), name);
    ordered_srclayers_.erase(iter);
  }
  void remove_dstlayers(string name){
    dstlayers_.erase(name);
    dstlayer_order_.erase(name);
    auto iter=ordered_dstlayers_.begin();
    while(iter!=ordered_dstlayers_.end()&&(*iter)->name()!=name)
      iter++;
    CHECK_EQ((*iter)->name(), name);
    ordered_dstlayers_.erase(iter);
  }
  void add_dstlayers(shared_ptr<Layer> layer){
    dstlayers_[layer->name()]=layer;
    dstlayer_order_[layer->name()]=dstid_++;
    ordered_dstlayers_.push_back(layer);
  }
  void add_srclayers(shared_ptr<Layer> layer){
    srclayers_[layer->name()]=layer;
    srclayer_order_[layer->name()]=srcid_++;
    ordered_srclayers_.push_back(layer);
  }


protected:
  string name_;
  //vector<shared_ptr<SyncedMem>> memblobs_;
  vector<vector<int>> shapes_;
  // DArray pos_, neg_;//for CD
  LayerProto layer_proto_;

  vector<shared_ptr<Layer>> ordered_dstlayers_, ordered_srclayers_;
  map<string, shared_ptr<Layer>> srclayers_, dstlayers_;
  // used to order src layers and dst layers
  map<string, int> srclayer_order_, dstlayer_order_;
  //<! current largest id for dst and src layers
  int dstid_, srcid_;
};

/****************************Middel Layers************************************/
/**
 * Multiply the col image with convolution weight, add bias to columns.
 */
class ConvolutionLayer: public Layer {
 public:
  virtual void Init(const LayerProto &proto);
  virtual void Setup(const vector<shared_ptr<Layer>>& src_layers);
  virtual void SetupAfterPartition(const vector<shared_ptr<Layer>>& src_layers);
  virtual void ComputeFeature(const vector<shared_ptr<Layer>>& src_layers);
  virtual void ComputeGradient(const vector<shared_ptr<Layer>>& src_layers);
  //virtual void CollectParams(vector<Param*> *params);
  //virtual vector<Param*> GetParams();

 protected:
  int kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_;
  int num_, channels_, height_,width_;
  //Param weight_, bias_;
};
class ConcateLayer: public Layer {
 public:
  virtual void Setup(const vector<shared_ptr<Layer>>& src_layers);
  virtual void SetupAfterPartition(const vector<shared_ptr<Layer>>& src_layers);
  virtual void ComputeFeature(const vector<shared_ptr<Layer>>& src_layers);
  virtual void ComputeGradient(const vector<shared_ptr<Layer>>& src_layers);
};

class ReLULayer: public Layer {
 public:
  virtual void Setup(const vector<shared_ptr<Layer>>& src_layers);
  virtual void ComputeFeature(const vector<shared_ptr<Layer>>& src_layers);
  virtual void ComputeGradient(const vector<shared_ptr<Layer>>& src_layers);
};

class DropoutLayer: public Layer {
 public:
  virtual void Setup(const vector<shared_ptr<Layer>>& src_layers);
  virtual void ComputeFeature(const vector<shared_ptr<Layer>>& src_layers);
  virtual void ComputeGradient(const vector<shared_ptr<Layer>>& src_layers);
  //virtual void Init(const LayerProto &proto);
  //virtual void ToProto(LayerProto *layer_proto, bool copyData);
 protected:
  float drop_prob_;
  /* record which neuron is dropped, required for back propagating gradients,
   * if mask[i]=0, then the i-th neuron is dropped.
   */
};

class InnerProductLayer: public Layer {
  /*
   * fully connected layer
   */
 public:
  virtual void Setup(const vector<shared_ptr<Layer>>& src_layers);
  virtual void SetupAfterPartition(const vector<shared_ptr<Layer>>& src_layers);
  virtual void ComputeFeature(const vector<shared_ptr<Layer>>& src_layers);
  virtual void ComputeGradient(const vector<shared_ptr<Layer>>& src_layers);
  //virtual void CollectParams(vector<Param*> *params);
  //virtual vector<Param*> GetParams();
  virtual void Init(const LayerProto &proto);
  //virtual void ToProto(LayerProto *layer_proto, bool copyData);
 private:
  //! dimension of the hidden layer
  int hdim_;
  //! dimension of the visible layer
  int vdim_;
  int num_;
  //Param weight_, bias_;
};
class NetSrcLayer: public Layer {
 public:
  virtual void ComputeFeature(const vector<shared_ptr<Layer>>& src_layers);
  virtual void ComputeGradient(const vector<shared_ptr<Layer>>& src_layers);

};
class NetDstLayer: public Layer {
 public:
  virtual void ComputeFeature(const vector<shared_ptr<Layer>>& src_layers);
  virtual void ComputeGradient(const vector<shared_ptr<Layer>>& src_layers);
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
  virtual void Init(const LayerProto &proto);
  virtual void ToProto(LayerProto *layer_proto, bool copyData);
  virtual void Setup(const vector<shared_ptr<Layer>>& src_layers);
  virtual void ComputeFeature(const vector<shared_ptr<Layer>>& src_layers);
  virtual void ComputeGradient(const vector<shared_ptr<Layer>>& src_layers);
 protected:
  //! shape of the bottom layer feature
  int num_, channels_, height_, width_;
  //! size local response (neighbor) area and padding size
  int size_, lpad_, rpad_;
  //! hyper-parameter
  float alpha_, beta_, knorm_;
  //DArray norm_, ratio_; //ratio : grad/(data*norm)
};
class PoolingLayer: public Layer {
 public:
  virtual void Init(const LayerProto &proto);
  virtual void ToProto(LayerProto *layer_proto, bool copyData);
  virtual void Setup(const vector<shared_ptr<Layer>>& src_layers);
  virtual void SetupAfterPartition(const vector<shared_ptr<Layer>>& src_layers);
  virtual void ComputeFeature(const vector<shared_ptr<Layer>>& src_layers);
  virtual void ComputeGradient(const vector<shared_ptr<Layer>>& src_layers);
 protected:
  int kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_;
  int num_,channels_, height_, width_, pooled_height_, pooled_width_;
};
class SliceLayer: public Layer {
 public:
  virtual const vector<int>& shapes(const Layer* layer) const;
  virtual void Setup(const vector<shared_ptr<Layer>>& src_layers);
  virtual void ComputeFeature(const vector<shared_ptr<Layer>>& src_layers);
  virtual void ComputeGradient(const vector<shared_ptr<Layer>>& src_layers);
};
class SplitLayer: public Layer {
 public:
  virtual void Init(const LayerProto &proto);
  virtual void Setup(const vector<shared_ptr<Layer>>& src_layers);
  virtual void SetupAfterPartition(const vector<shared_ptr<Layer>>& src_layers);
  virtual void ComputeFeature(const vector<shared_ptr<Layer>>& src_layers);
  virtual void ComputeGradient(const vector<shared_ptr<Layer>>& src_layers);

};

/**
 * This layer apply Tan function to neuron activations.
 * f(x)=A tanh(Bx)
 * f'(x)=B/A (A*A-f(x)*f(x))
 */
class TanhLayer: public Layer {
 public:
  virtual void Setup(const vector<shared_ptr<Layer>>& src_layers);
  virtual void ComputeFeature(const vector<shared_ptr<Layer>>& src_layers);
  virtual void ComputeGradient(const vector<shared_ptr<Layer>>& src_layers);
};

/**********************Output Performance/Loss Layers************************/
const int kLoss=1;
const int kPrecision=2;

class PerformanceLayer: public Layer{
 public:
  virtual Performance ComputePerformance(const vector<shared_ptr<Layer>>&src_layers,
      int type)=0;
};
class SoftmaxLossLayer: public PerformanceLayer {
  /*
   * connected from the label layer and the last fc layer
   */
 public:
  virtual void Setup(const vector<shared_ptr<Layer>>& src_layers);
  virtual void ComputeFeature(const vector<shared_ptr<Layer>>& src_layers);
  virtual void ComputeGradient(const vector<shared_ptr<Layer>>& src_layers);
  virtual Performance ComputePerformance(const vector<shared_ptr<Layer>>&src_layers,
      int type);
 private:
  int num_;
  int dim_;
  int top_k_;
};
/***********************Inpute Layers****************************************/
class InputLayer: public Layer {
 public:
  virtual bool HasInput() { return true; }
  virtual void AddInputRecord(const Record& record, Phase phase=kTrain)=0;
  //virtual void SetInputData(DArray *data);
  virtual void ComputeFeature(const vector<shared_ptr<Layer>>& src_layers){};
  virtual void ComputeGradient(const vector<shared_ptr<Layer>>& src_layers){};
  virtual void Setup(const vector<vector<int>>& shapes)=0;
  virtual void Setup(const int batchsize, const Record & record);
  virtual void Setup(const vector<shared_ptr<Layer>>& src_layers);
  virtual void SetupAfterPartition(const vector<shared_ptr<Layer>>& src_layers);

  //DArray* mutable_prefetch_data(){return &(this->grad_);}
  //DArray* mutable_grad(){return nullptr;}
 protected:
  //DArray prefetch_data_; use the grad_ field for prefetch data
  int offset_;
};
class RGBImageLayer: public InputLayer {
 public:
  virtual void Setup(const vector<shared_ptr<Layer>>& src_layers);
  virtual void SetupAfterPartition(const vector<shared_ptr<Layer>>& src_layers);
  virtual void Setup(const vector<vector<int>>& shapes);
  virtual void Setup(const int batchsize, const Record & record);
  void Setup(const vector<int>& shape);
  virtual void AddInputRecord(const Record& record, Phase phase=kTrain);

 private:
  bool mirror_;
  int cropsize_;
  float scale_;
};
class LabelLayer: public InputLayer {
 public:
  virtual void Setup(const vector<vector<int>>& shapes);
  virtual void Setup(const int batchsize, const Record & record);
  virtual void Setup(const vector<shared_ptr<Layer>>& src_layers);
  virtual void SetupAfterPartition(const vector<shared_ptr<Layer>>& src_layers);

  virtual void AddInputRecord(const Record& record, Phase phase=kTrain);
};
class MnistImageLayer: public InputLayer {
 public:
  typedef std::uniform_real_distribution<float> UniformDist;
  virtual void Setup(const vector<vector<int>>& shapes);
  virtual void Setup(const int batchsize, const Record & record);
  virtual void Setup(const vector<shared_ptr<Layer>>& src_layers);
  virtual void SetupAfterPartition(const vector<shared_ptr<Layer>>& src_layers);
  void Setup(const vector<int>& shape);

  virtual void AddInputRecord(const Record& record, Phase phase=kTrain);
  virtual ~MnistImageLayer();
  vector<uint8_t> Convert2Image(int k);
 protected:
  std::default_random_engine generator_;
  // height and width of the image after deformation
  int h_,w_;
  // kernel size for elastic distortion
  int kernel_;
  // n^2 images are processed as a batch for elastic distortion
  int n_;
  // conv height and conv width
  int conv_h_, conv_w_;
  // gauss kernel values, displacements, column image and tmp buffer
  float* gauss_, *displacementx_, *displacementy_, *colimg_, *tmpimg_;
};
}  // namespace lapis

#endif  // INCLUDE_NET_LAYER_H_
