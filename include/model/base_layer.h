#ifndef INCLUDE_BASE_LAYER_H_
#define INCLUDE_BASE_LAYER_H_

#include <vector>
#include <string>
#include <map>
#include <functional>
#include <utility>
#include <memory>
#include <chrono>
#include <random>
#include <algorithm>

#include "proto/model.pb.h"
#include "model/param.h"
#include "utils/common.h"
#include "utils/blob.h"

using std::vector;
using std::shared_ptr;
using std::make_shared;
using std::string;
using std::map;

namespace singa{

class Layer;
typedef shared_ptr<Layer> SLayer;
/**
 * Base layer class.
 * Children should implement at least Layer::Setup, Layer::ComputeFeature(),
 * Layer::ComputGradient() functions for backpropagation method;
 * TODO(wangwei) implement children layers to support contrastive divergence,
 * The identifier of each layer is the literal string of the class name without
 * the suffix "Layer", which is used in layer registration and creation.
 */
class Layer {
 public:
  Layer(){}
  /**
   * simply save the proto configuation.
   * most initializations are done by Setup().
   * @param layer_proto user defined layer configuration
   */
  virtual void Init(const LayerProto &proto);
  /**
   * copy layer configuration from the other Layer, and set the shape.
   */
  void Init(const Layer& other, const vector<int>& shape);
  virtual ~Layer(){}
  /**
   * Marshal layer properties and data into google protobuf object
   * (i.e., snapshot).
   * Parameters are marshalled separately into another object (i.e., model).
   * @param layer_proto
   * @param copyData if true marshal data of DArray
   */
  virtual void ToProto(LayerProto *layer_proto, bool copyData);
  /**
   * Setup layer properties.
   * Setup the shapes for data and parameters, also setup some properties
   * based on the layer configuration and connected src layers.
   * @param srclayers layers connecting to this layer
   */
  virtual void Setup(const LayerProto& proto,
      const vector<SLayer>& srclayers)=0;
  /**
   * \copydoc Setup(const LayerProto&, const vector<SLayer>&)
   */
  virtual void Setup();
  /**
   * Setup the layer properties except shape.
   * the shape is already set and passed in to set other properties.
   * perperties are set according to shapes of itself and connected layers, and
   * configuration. this should not change the current shape_(
   * shape check is done outside the function).
   */
  virtual void SetupAfterPartition(const LayerProto& proto,
      const vector<int> &shape,
      const vector<SLayer>& srclayers)=0;
  /**
   * \copybrief SetupAfterPartition(const LayerProto&, const vector<int> &,
   * const vector<SLayer>& ).
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
   */
  virtual vector<Param*> GetParams(){ return vector<Param*>(); }
  /**
   * Compute features of this layer based on connected layers.
   * Implement forward propagation for BP; TODO Implement both postive phase
   * and negative phase for CD.
   * @param srclayers layers connecting to this layer
   */
  virtual void ComputeFeature(const vector<SLayer>& srclayers)=0;
  /**
   * \copybrief ComputeFeature(const vector<SLayer>& srclayers)
   */
  virtual void ComputeFeature();
  /**
   * Compute gradients for parameters and connecting layers.
   * Implement backward propagation for BP; TODO Calculate gradients for
   * parameters for CD.
   * @param srclayers layers connecting to this layer.
   */
  virtual void ComputeGradient(const vector<SLayer>& srclayers)=0;
  /**
   * \copybrief ComputeGradient(const vector<SLayer>& srclayers)
   */
  virtual void ComputeGradient();
  /**
   * decide on which dimension of DArray to do the partitioning.
   * @mode kModel, kData, kHybrid, kNone (no partition)
   * @return the partition dimension, -1 for no partition
   */
  virtual int partition_dimension() const {
    int ret=0;
    if(partition_type()==kLayerPartition)
      ret= 1;
    else if(partition_type()==kNone)
      ret= -1;
    return ret;
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
  virtual const vector<int>& shape(const Layer* layer=nullptr) const{
    return data_.shape();
  }

  /**
   * @return a const ref for Blob storing neuron values of this layer for BP
   */
  virtual const Blob<float>& data(int k=0){
    return data_;
  }
  virtual Blob<float>* mutable_data(int k=0){
    return &data_;
  }

  /**
   * @return a pointer to storing neuron grads of this layer for BP
   */
  virtual Blob<float>* mutable_grad(int k=0) {
    return &grad_;
  }

  virtual const vector< SLayer> srclayers() const {
    return srclayers_;
  }
  virtual const vector<SLayer> dstlayers() const {
    return dstlayers_;
  }

  virtual const int srclayers_size() const {
    return srclayers_.size();
  }
  virtual const int dstlayers_size() const {
    return dstlayers_.size();
  }
  virtual void ClearDstLayers() {
    dstlayers_.clear();
  }
  virtual void ClearSrcLayers() {
    srclayers_.clear();
  }

  virtual void AddSrcLayer(SLayer src){
    srclayers_.push_back(src);
  }
  virtual void AddDstLayer(SLayer dst){
    dstlayers_.push_back(dst);
  }

protected:
  string name_;
  //vector<shared_ptr<SyncedMem>> memblobs_;
  Blob<float> data_, grad_;
  // DArray pos_, neg_;//for CD
  LayerProto layer_proto_;
  vector<SLayer> srclayers_, dstlayers_;
};

class BridgeSrcLayer: public Layer {
 public:
  virtual void Setup(const LayerProto& proto, const vector<SLayer>& srclayers);
  virtual void SetupAfterPartition();
  virtual void SetupAfterPartition(const LayerProto& proto,
      const vector<int> &shape,
      const vector<SLayer>& srclayers){}

  virtual void ComputeFeature(const vector<SLayer>& srclayers);
  virtual void ComputeGradient(const vector<SLayer>& srclayers);
};
class BridgeDstLayer: public Layer {
 public:
  virtual void Setup(const LayerProto& proto, const vector<SLayer>& srclayers);
  virtual void SetupAfterPartition();
  virtual void SetupAfterPartition(const LayerProto& proto,
      const vector<int> &shape,
      const vector<SLayer>& srclayers){}

  virtual void ComputeFeature(const vector<SLayer>& srclayers);
  virtual void ComputeGradient(const vector<SLayer>& srclayers);
};
class ConcateLayer: public Layer {
 public:
  virtual void Setup(const LayerProto& proto, const vector<SLayer>& srclayers);
  virtual void SetupAfterPartition();
  virtual void SetupAfterPartition(const LayerProto& proto,
      const vector<int> &shape,
      const vector<SLayer>& srclayers){}

  virtual void ComputeFeature(const vector<shared_ptr<Layer>>& srclayers);
  virtual void ComputeGradient(const vector<shared_ptr<Layer>>& srclayers);
};


/**
 * base layer for prefetching records from local Shard, HDFS, lmdb, etc.
 * cannot be partitioned, always returns kNone for partition type.
 */

class DataLayer: public Layer{
 public:
  virtual void ComputeFeature(const vector<SLayer>& srclayers)=0;
  virtual void Setup(const LayerProto& proto, const vector<SLayer>& srclayers)=0;
  virtual void ComputeGradient(const vector<SLayer>& srclayers){};
  virtual const vector<Record>& records() const {
    return records_;
  }
  virtual void Setup(){
    vector<SLayer> dummy;
    Setup(layer_proto_,dummy);
    has_set_=true;
  }
  virtual void SetupAfterPartition(const LayerProto& proto,
      const vector<int> &shape,
      const vector<SLayer>& srclayers){}

  virtual void SetupAfterPartition(){
    if(!has_set_)
    Setup();
  }
  virtual PartitionType partition_type () const {
    return kNone;
  }

  virtual int batchsize() const {
    return layer_proto_.data_param().batchsize();
  }
  virtual const Record& sample() const {
    return sample_;
  }

  virtual void CompletePrefetch(){
    records_.swap(prefetch_data_);
  }

 protected:
  bool has_set_;
  int random_skip_, batchsize_;
  Record sample_;
  vector<Record> records_, prefetch_data_;
};
class SliceLayer: public Layer {
 public:
  virtual void Setup(const LayerProto& proto, const vector<SLayer>& srclayers);
  virtual void SetupAfterPartition();
  virtual void SetupAfterPartition(const LayerProto& proto,
      const vector<int> &shape,
      const vector<SLayer>& srclayers){}


  virtual const vector<int>& shape(const Layer* layer=nullptr) const;
  virtual void ComputeFeature(const vector<shared_ptr<Layer>>& srclayers);
  virtual void ComputeGradient(const vector<shared_ptr<Layer>>& srclayers);

 protected:
  vector<vector<int>> shapes_;
};


class SplitLayer: public Layer {
 public:
  virtual void Setup(const LayerProto& proto, const vector<SLayer>& srclayers);
  virtual void SetupAfterPartition();
  virtual void SetupAfterPartition(const LayerProto& proto,
      const vector<int> &shape,
      const vector<SLayer>& srclayers){}

  virtual void ComputeFeature(const vector<shared_ptr<Layer>>& srclayers);
  virtual void ComputeGradient(const vector<shared_ptr<Layer>>& srclayers);
};
/**********************Output Performance/Loss Layers************************/
const int kLoss=1;
const int kPrecision=2;

class PerformanceLayer: public Layer{
 public:
  virtual Performance ComputePerformance(
      const vector<SLayer>&srclayers,
      int type)=0;
};

/**
 * parse the input blob/record into meaning full format.
 */
class ParserLayer: public Layer {
 public:
  virtual void ComputeFeature(const vector<SLayer>& srclayers)=0;
  virtual void Setup(const LayerProto& proto, const vector<SLayer>& srclayers)=0;
  virtual void ComputeGradient(const vector<SLayer>& srclayers){};
  virtual void Setup(){
    Setup(layer_proto_,srclayers_);
    has_set_=true;
  }
  virtual void SetupAfterPartition(){
    if(!has_set_)
      Setup();
  }
  virtual void SetupAfterPartition(const LayerProto& proto,
      const vector<int> &shape,
      const vector<SLayer>& srclayers){}

  virtual PartitionType partition_type () {
    return kNone;
  }

  virtual Blob<float>* mutable_grad(int k=0) {
    NOT_IMPLEMENTED;
    return &grad_;
  }

 private:
  bool has_set_;
};
} // singa

#endif // INCLUDE_BASE_LAYER_H_
