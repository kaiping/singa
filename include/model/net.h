#ifndef INCLUDE_NET_NET_H_
#define INCLUDE_NET_NET_H_

#include <glog/logging.h>
#include <vector>
#include <map>
#include <unordered_set>
#include <stack>
#include "model/param.h"
#include "proto/model.pb.h"
#include "model/layer.h"

namespace singa {
/**
 * The neural network is constructed from user configured layers through google
 * protocol buffer or by adding layers explicitly.
 *
 * Some layers, e.g., SplitLayer and NetSrcLayer/NetDstLayer  will be added
 * implicitly.
 */
class NeuralNet {
 public:
  /**
   * construct the net structure from protocol buffer.
   */
  explicit Net(const NetProto &net_proto);
  /**
   * desctruct the net.
   * free layer objects.
   */
  ~Net();
  /**
   * construct a string for describing the layers and parameters, including
   * shape info.
   */
  std::string ToString();
  /**
   * print the DOT string for drawing a graph for the neural net
   */
  std::string ToDOTString();
  /**
   * Add layer explicitly used in manually programming/constructing neural net.
   */
  void AddLayer(const LayerProto &layer_proto);
  /**
   * Add layer explicitly used in manually programming/constructing neural net.
   */
  void AddLayer(const Layer* layer);

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
    const std::vector<Layer *>& layers() {
    return layers_;
  }
  const std::vector<Param *> &params() {
    return params_;
  }
  Layer* name2layer(string name){
    if (name2layer_.find(name)!=name2layer_.end())
      return name2layer_[name];
    else return NULL;
  }
  const vector<Layer*> name2srclayers(string name){
     if (name2srclayers_.find(name)!=name2srclayers_.end())
      return name2srclayers_[name];
    else return vector<Layer*>{};
  }
  const vector<Layer*> name2dstlayers(string name){
    if (name2dstlayers_.find(name)!=name2dstlayers_.end())
      return name2dstlayers_[name];
    else return vector<Layer*>{};
  }

 protected:
  void check();
  /**
   * called internally to setup the neural net without considering partitions.
   * the input layers are setup
   */
  void Setup();
  /**
   * Partition each layer according its partition type and dimension.
   * @param layers original unpartitioned layers
   */
  map<string, vector<shared_ptr<Layer>> PartitionNeuralNet(
      const vector<shared_ptr<Layer>>& layers);
  /**
   * connect partitioned layers by adding helper layers, e.g., ConcateLayer
   * and SliceLayer.
   * TODO distinguish kOneToMany from kOneToOne. Now kOnetoMany is
   * processed the same as kOneToOne.
   */
  void ConnectPartitionedLayers(
      const map<string, vector<shared_ptr<Layer>>>& partitioned_layers,
      const map<pair<string,string>, ConnectionType>& connections,
      vector<shared_ptr<Layer>* layers,
      map<string, shared_ptr<Layer>* name2srclayers,
      map<string, shared_ptr<Layer>* name2dstlayers);

  /**
   * Add SliceLayer to connect src_layer and dst_layers.
   */
  void AddSliceLayer(int slice_dimension, shared_ptr<Layer> src_layer,
    vector<shared_ptr<Layer>> dst_layers, vector<shared_ptr<Layer>> *layers,
    map<string, shared_ptr<Layer>* name2srclayers);
  /**
   * add ConcateLayer to connect src_layers and dst_layer
   */
  void AddConcateLayer(int concate_dimension,
    vector<shared_ptr<Layer>> src_layers,
    shared_ptr<Layer> dst_layer, vector<shared_ptr<Layer>> *layers,
    map<string, shared_ptr<Layer>* name2srclayers);
  /**
   * add a split layer for the layer which has multiple outgoing connected
   * layers (exception SliceLayer).
   */
  void AddSplitLayers(
    vector<shared_ptr<Layer>> *layers,
    map<string, vector<shared_ptr<Layer>>> *name2dstlayers);
  /**
   * add a NetSrcLayer and NetDstLayer between any connection whose ending
   * layers resident on different machines.
   */
  void AddNetTransferLayers(
    vector<shared_ptr<Layer>> *layers,
    map<string, vector<shared_ptr<Layer>>> *name2dstlayers);

  // SortLayersForBP
  void topology_sort(vector<Layer *> *layers,
                     const map<string, vector<Layer*>>& name2dstlayers);
  void topology_sort_inner(Layer *layer,
                         const std::map<Layer *,
                         std::vector<Layer *>> &adjacent_list,
                         std::map<Layer *, bool> *visited,
                         std::stack<Layer *> *stack) ;
  // TODO SortLayersForCD
 private:
  std::vector<Layer *> layers_;
  std::vector<PerformanceLayer *> performance_layers_;
  std::vector<InputLayer *> input_layers_;
  std::vector<Param *> params_;

  std::map<string, Layer*> name2layer_;
  std::map<string, vector<Layer*>> name2srclayers_;;
  std::map<string, vector<Layer*>> name2dstlayers_;;
};
}  // namespace singa
#endif  // INCLUDE_NET_NET_H_
