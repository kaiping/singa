#ifndef INCLUDE_NET_NET_H_
#define INCLUDE_NET_NET_H_

#include <glog/logging.h>
#include <vector>
#include <map>
#include <memory>

//#include "model/param.h"
#include "proto/model.pb.h"
#include "model/layer.h"
#include "utils/cluster.h"
#include "utils/factory.h"
#include "utils/graph.h"

using std::vector;
using std::string;
using std::map;
using std::shared_ptr;
namespace singa {
/**
 * The neural network is constructed from user configured layers through google
 * protocol buffer. TODO support constructing neural network by adding layers
 * explicitly. E.g., users create layers and connect them manually in the code.
 *
 * Some layers, e.g., SplitLayer and NetSrcLayer/NetDstLayer will be added
 * implicitly to partition the neural network.
 */
class NeuralNet {
 public:
  /**
   * construct the net structure from protocol buffer.
   */
  explicit NeuralNet(const NetProto &net_proto);
  void Init(const NetProto &net_proto, const shared_ptr<Cluster>& cluster) ;
  /**
   * desctruct the net.
   * free layer objects.
   */
  ~NeuralNet();
  /**
   * construct a string for describing the layers and parameters, including
   * shape info.
   */
  std::string ToString();
  /**
   * print the DOT string for drawing a graph for the neural net
   */
  void DisplayNeuralNet(const vector<shared_ptr<Layer>>& layers);

  /**
   * Add layer explicitly used in manually programming/constructing neural net.
   */
  void AddLayer(const LayerProto &layer_proto){};
  /**
   * Add layer explicitly used in manually programming/constructing neural net.
   */
  void AddLayer(const Layer* layer){};
  /**
   * set the meta data of layer, e.g., shape.
   * shapes of the first layer are infered from input records/shapes.
   * Memory is not allocated until first time used.
   *
   * @input_shapes shapes for the input layers
   */
  void Setup(const vector<vector<int>>& input_shapes);
  /**
   * @batchsize mini-batch size
   * @record input record to the net, used to set the shapes of input layers
   */
  void Setup(int batchsize, const Record &record);
  /**
   * called internally to setup the neural net without considering partitions.
   * the input layers' shapes are from google protobuf config
   */
  void Setup();

  /**
   * serialize the net.
   */
  void ToProto(NetProto *net_proto, bool copyData=false);
  PerformanceLayer* performance_layer(int k) {
    CHECK_LT(k, performance_layers_.size());
    return performance_layers_[k];
  }
  const std::vector<InputLayer *> &input_layer() {
    return input_layers_;
  }
  InputLayer * input_layer(int k) {
    CHECK_LT(k, input_layers_.size());
    return input_layers_[k];
  }
  const std::vector<shared_ptr<Layer>>& layers() {
    return layers_;
  }
    /*
  const std::vector<Param *> &params() {
    return params_;
  }
  */
  shared_ptr<Layer> name2layer(string name){
    if (name2layer_.find(name)!=name2layer_.end())
      return name2layer_[name];
    else return NULL;
  }

 protected:
  void ConstructNeuralNet(const NetProto &net_proto);
  void PartitionNeuralNet();
  map<string, shared_ptr<Layer>> GetNameToLayer(
    const vector<shared_ptr<Layer>>& layers);
  Graph CreatePartitonedGraph(const vector<shared_ptr<Layer>>& layers,
    const map<string, shared_ptr<Layer>>& name2layer);

  /**
   * Partition each layer according its partition type and dimension.
   * @param layers original unpartitioned layers
   */
  map<string, vector<shared_ptr<Layer>>> PartitionLayers(
      const vector<shared_ptr<Layer>>& layers);
  /**
   * connect partitioned layers by adding helper layers, e.g., ConcateLayer
   * and SliceLayer.
   * TODO distinguish kOneToMany from kOneToOne. Now kOnetoMany is
   * processed the same as kOneToOne.
  vector<shared_ptr<Layer>> ConnectPartitionedLayers(
      const map<string, vector<shared_ptr<Layer>>>& partitioned_layers,
      const vector<shared_ptr<Layer>>& layers);
   */

  /**
   * Add SliceLayer to connect src_layer and dst_layers.
  void InsertSliceLayer(const int slice_dimension, shared_ptr<Layer> src_layer,
    const vector<shared_ptr<Layer>> dst_layers,
    vector<shared_ptr<Layer>> *layers);
   */
  /**
   * add ConcateLayer to connect src_layers and dst_layer
  void InsertConcateLayer(const int concate_dimension,
    const vector<shared_ptr<Layer>>& src_layers,
    shared_ptr<Layer> dst_layer, vector<shared_ptr<Layer>> *layers);
   */
  /**
   * add a split layer for the layer which has multiple outgoing connected
   * layers (exception SliceLayer).
 vector<shared_ptr<Layer>> InsertSplitLayers(
     const vector<shared_ptr<Layer>> &layers);
   */
  /**
   * add a NetSrcLayer and NetDstLayer between any connection whose ending
   * layers resident on different machines.
 vector<shared_ptr<Layer>> InsertNetTransferLayers(
     const vector<shared_ptr<Layer>> &layers);
   */
 private:
  vector<shared_ptr<Layer>> layers_;
  vector<PerformanceLayer *> performance_layers_;
  vector<InputLayer *> input_layers_;
  //vector<shared_ptr<Param>> params_;
  map<string, shared_ptr<Layer>> name2layer_;
  shared_ptr<Cluster> cluster_;
  Factory<Layer>* factory_;
};
}  // namespace singa
#endif  // INCLUDE_NET_NET_H_
