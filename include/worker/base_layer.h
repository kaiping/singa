#ifndef INCLUDE_BASE_LAYER_H_
#define INCLUDE_BASE_LAYER_H_

#include <vector>
#include <string>
#include <map>
#include <functional>
#include <utility>
#include <memory>
#include <chrono>
#include <algorithm>

#include "proto/model.pb.h"
#include "utils/param.h"
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
  virtual void ComputeFeature(bool training, const vector<SLayer>& srclayers)=0;
  /**
   * \copybrief ComputeFeature(const vector<SLayer>& srclayers)
   */
  virtual void ComputeFeature(bool training);
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
  const vector<int>& shape(const Layer* layer=nullptr) const{
    return data(layer).shape();
  }

  /**
   * @return a const ref for Blob storing neuron values of this layer for BP
   */
  virtual const Blob<float>& data(const Layer* from=nullptr) const {
    return data_;
  }
  virtual Blob<float>* mutable_data(const Layer* from=nullptr){
    return &data_;
  }

  virtual const Blob<float>& grad(const Layer* from=nullptr) const {
    return grad_;
  }
  /**
   * @return a pointer to storing neuron grads of this layer for BP
   */
  virtual Blob<float>* mutable_grad(const Layer* from=nullptr) {
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

  virtual bool is_datalayer() const {
    return false;
  }
  virtual bool is_parserlayer() const {
    return false;
  }
  virtual bool is_losslayer() const {
    return false;
  }
  virtual bool is_bridgesrclayer() const {
    return false;
  }
  virtual bool is_bridgedstlayer() const {
    return false;
  }
  virtual void set_ready(bool a) const{
  }
  virtual bool ready() const{
    return true;
  }

  /*
  virtual bool is_neuronlayer() const {
    return false;
  }
  */

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

  virtual void ComputeFeature(bool training, const vector<SLayer>& srclayers);
  virtual void ComputeGradient(const vector<SLayer>& srclayers);
  virtual bool is_bridgesrclayer() const {
    true;
  }

  virtual void set_ready(bool a) const {
    ready_=a;
  }
  virtual bool ready() const {
    return ready_;
  }
 protected:
  bool ready_;
};
class BridgeDstLayer: public Layer {
 public:
  virtual void Setup(const LayerProto& proto, const vector<SLayer>& srclayers);
  virtual void SetupAfterPartition();
  virtual void SetupAfterPartition(const LayerProto& proto,
      const vector<int> &shape,
      const vector<SLayer>& srclayers){}

  virtual void ComputeFeature(bool training, const vector<SLayer>& srclayers);
  virtual void ComputeGradient(const vector<SLayer>& srclayers);
  virtual bool is_bridgedstlayer() const {
    true;
  }
  virtual void set_ready(bool a) const {
    ready_=a;
  }
  virtual bool ready() const {
    return ready_;
  }
 protected:
  bool ready_;
};
class ConcateLayer: public Layer {
 public:
  virtual void Setup(const LayerProto& proto, const vector<SLayer>& srclayers);
  virtual void SetupAfterPartition();
  virtual void SetupAfterPartition(const LayerProto& proto,
      const vector<int> &shape,
      const vector<SLayer>& srclayers){}

  virtual void ComputeFeature(bool training, const vector<shared_ptr<Layer>>& srclayers);
  virtual void ComputeGradient(const vector<shared_ptr<Layer>>& srclayers);
};


/**
 * base layer for prefetching records from local Shard, HDFS, lmdb, etc.
 * cannot be partitioned, always returns kNone for partition type.
 */

class DataLayer: public Layer{
 public:
  virtual void ComputeFeature(bool training, const vector<SLayer>& srclayers)=0;
  virtual void Setup(const LayerProto& proto, const vector<SLayer>& srclayers)=0;
  virtual bool is_datalayer() const {
    return true;
  }
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

 protected:
  bool has_set_;
  int random_skip_, batchsize_;
  Record sample_;
  vector<Record> records_;
};
class SliceLayer: public Layer {
 public:
  virtual void Setup(const LayerProto& proto, const vector<SLayer>& srclayers);
  virtual void SetupAfterPartition();
  virtual void SetupAfterPartition(const LayerProto& proto,
      const vector<int> &shape,
      const vector<SLayer>& srclayers){}


  virtual const Blob<float>& data(const Layer* layer=nullptr) const;
  virtual const Blob<float>& grad(const Layer* layer=nullptr) const;
  virtual Blob<float>* mutable_data(const Layer* layer=nullptr);
  virtual Blob<float>* mutable_grad(const Layer* layer=nullptr);
  virtual void ComputeFeature(bool training, const vector<shared_ptr<Layer>>& srclayers);
  virtual void ComputeGradient(const vector<shared_ptr<Layer>>& srclayers);

 protected:
  int SliceID(const Layer* layer) const;
  vector<Blob<float>> datavec_, gradvec_;
};


class SplitLayer: public Layer {
 public:
  virtual void Setup(const LayerProto& proto, const vector<SLayer>& srclayers);
  virtual void SetupAfterPartition();
  virtual void SetupAfterPartition(const LayerProto& proto,
      const vector<int> &shape,
      const vector<SLayer>& srclayers){}

  virtual void ComputeFeature(bool training, const vector<shared_ptr<Layer>>& srclayers);
  virtual void ComputeGradient(const vector<shared_ptr<Layer>>& srclayers);
};
/**********************Loss Layers************************/

class LossLayer: public Layer{
 public:
  virtual void Setup(const LayerProto& proto,
      const vector<SLayer>& srclayers)=0;

  virtual void SetupAfterPartition(const LayerProto& proto,
      const vector<int> &shape,
      const vector<SLayer>& srclayers)=0;
  virtual Blob<float>* mutable_grad(Layer* layer=nullptr){
    return nullptr;
  }
  virtual const Blob<float>& grad(const Layer* from=nullptr) const {
    CHECK(false)<<"Loss layer has not gradient blob";
    return grad_;
  }
  virtual bool is_losslayer() const {
    return true;
  }

  virtual const Blob<float>& metric() const {
    return metric_;
  }
 protected:
  Blob<float> metric_;
};

/**
 * parse the input blob/record into meaning full format.
 */
class ParserLayer: public Layer {
 public:
  virtual void ComputeFeature(bool training);
  virtual void ComputeFeature(bool training, const vector<SLayer>& srclayers)=0;
  virtual void Setup(const LayerProto& proto, const vector<SLayer>& srclayers)=0;
  virtual bool is_parserlayer() const {
    return true;
  }
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

  virtual PartitionType partition_type () const{
    return kNone;
  }

  virtual Blob<float>* mutable_grad(const Layer* layer=nullptr) {
    return nullptr;
  }
  virtual const Blob<float>& grad(const Layer* from=nullptr) const {
    CHECK(false)<<"Parser layer has not gradient blob";
    return grad_;
  }

  /**
   * prefetching is transparent to parsing logics.
   * users implement parsing logics in ComputeFeature(const vector<SLayer>&)
   * worker/training algorithm calls this function to do prefetching in a
   * thread. data is in fact parsed into prefetch_data_.
   */
  void Prefetching(bool training){
    if(prefetch_data_.count()==0)
      prefetch_data_.ReshapeLike(data_);
    data_.Swap(prefetch_data_);
    ComputeFeature(training, srclayers_);
  }

  /**
   * must be called after Prefetching and before calling upper layers
   * otherwise, the old data will be used (new data is in prefech_data_).
   */
  void CompletePrefetching(){
    data_.Swap(prefetch_data_);
  }

  /**
   * if prefetching, then do nothing; otherwise conduct normal ComputeFeature
   */
  void ComputeFeature(bool training){
    if(prefetch_data_.count()==0)
      ComputeFeature(training, srclayers_);
  }

 private:
  bool has_set_;
  //!< prefetch_data_ is invisible to layer logics, i.e., parsing.
  Blob<float> prefetch_data_;
};
} // singa

#endif // INCLUDE_BASE_LAYER_H_
