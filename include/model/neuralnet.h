#ifndef INCLUDE_NET_NET_H_
#define INCLUDE_NET_NET_H_

#include <glog/logging.h>
#include <vector>
#include <map>
#include <memory>

//#include "model/param.h"
#include "proto/model.pb.h"
#include "model/layer.h"
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
  NeuralNet(NetProto net_proto, int group_size=1);
  /**
   * construct a string for describing the layers and parameters, including
   * shape info.
   */
  std::string ToString();

  std::string ToAdjacency();
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
   * serialize the net.
   */
  void ToProto(NetProto *net_proto, bool copyData=false);
  PerformanceLayer* performance_layer(int k) {
    CHECK_LT(k, performance_layers_.size());
    return performance_layers_[k];
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
  //vector<shared_ptr<Param>> params_;
  map<string, shared_ptr<Layer>> name2layer_;
  map<string, LayerProto> name2layerproto_;
  Factory<Layer>* factory_;
  int group_size_;
  Graph graph_;
};
}  // namespace singa
#endif  // INCLUDE_NET_NET_H_
