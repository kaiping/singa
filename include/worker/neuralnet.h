#ifndef INCLUDE_NET_NET_H_
#define INCLUDE_NET_NET_H_

#include <glog/logging.h>
#include <vector>
#include <map>
#include <memory>

#include "proto/model.pb.h"
#include "worker/layer.h"
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
  string DebugInfo();

  std::string ToAdjacency();
  /**
   * print the DOT string for drawing a graph for the neural net
   */
  void DisplayNeuralNet(const vector<shared_ptr<Layer>>& layers);
/**
   * Print Norm1 of data and grad of each Layer and parameter.
   * @param net, neural network
   */

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
  void ShareWeights(shared_ptr<NeuralNet> net){}
  void ToProto(NetProto *net_proto, bool copyData=false);
  const std::vector<shared_ptr<Layer>>& layers() {
    return layers_;
  }
  const std::vector<ParserLayer*>& parserlayers() {
    if(parserlayers_.size()==0){
      for(auto& layer: layers_)
        if(layer->is_parserlayer())
          parserlayers_.push_back(static_cast<ParserLayer*>(layer.get()));
    }
    return parserlayers_;
  }
  const std::vector<LossLayer*>& losslayers() {
    if(losslayers_.size()==0){
      for(auto& layer: layers_)
        if(layer->is_losslayer())
          losslayers_.push_back(static_cast<LossLayer*>(layer.get()));
    }
    return losslayers_;
  }
  const std::vector<DataLayer*>& datalayers() {
    if(datalayers_.size()==0){
      for(auto& layer: layers_)
        if(layer->is_datalayer())
          datalayers_.push_back(static_cast<DataLayer*>(layer.get()));
    }
    return datalayers_;
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

  Param* paramid2param(int id) {
    if(paramid2param_.size()==0){
      for(auto& layer: layers_){
        for(Param* p: layer->GetParams()){
          paramid2param_[p->id()]=p;
        }
      }
    }
    return paramid2param_[id];
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
  vector<ParserLayer*> parserlayers_;
  vector<LossLayer*> losslayers_;
  vector<DataLayer*> datalayers_;
  vector<Param*> params_;
  map<string, shared_ptr<Layer>> name2layer_;
  map<int, Param*> paramid2param_;
  map<string, LayerProto> name2layerproto_;
  Factory<Layer>* factory_;
  int group_size_;
  Graph graph_;
};
}  // namespace singa
#endif  // INCLUDE_NET_NET_H_
