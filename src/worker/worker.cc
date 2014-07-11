// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-04 11:49

#include "worker/worker.h"

using std::vector;

namespace lapis {
Worker::Worker(const DistributedMemory* dm,
               const DistributedDisk* dd,
               const GlobalContext* gc) {
  model_controller_ = new ModelController(dm, dd, gc);
}

void Worker::CreateNet(const LayerProto& layers_proto,
                       vector<Layer*>* layers,
                       vector<Edge*>* edges) {
  for (auto& layer_proto : layers_proto) {
    Layer* layer = layer_factory_get(layer_proto.name());
    layer->init(layer_proto, edges);
    layers->push_back(layer);
  }
}

bool HasIntersection(const vector<Blob*>& A, const vector<Blob*>& B) {
  for (Blob* a : A)
    for (Blob* b : B)
      if (a == b)
        return true;
  return false;
}

void topological_sort_inner(const Layer* layer,
                      const std::map<Layer*, vector<Layer*>>& adjacent_list,
                      std::map<Layer*, bool>* visited,
                      std::stack<Layer*>* stack) {
  (*visited)[layer] = true;
  for (Layer* layer1 : adjacent_list[layer]) {
    if ((*visited)[layer])
      continue;
    topological_sort_inner(layer1, adjacent_list, visited, stack);
  }
  stack->push(layer);
}

void topological_sort(vector<Layer*>* layers) {
  // adjacent list from upper layers to lower layers
  std::map<Layer*, vector<Layer*>> adjacent_list;
  std::map<Layer*, bool> visited;
  for (Layer* layer : layers) {
    adjacent_list[layer];  // will automatically insert a new entry
    visited[layer] = false;
    for (Layer* layer1 : layers) {
      if (layer == layer1)
        continue;
      if (HasIntersection(layer.lower_blobs_, layer1.upper_blobs_))
        adjacent_list[layer].push_back(layer1);
    }
  }
  std::stack<Layer*> stack;
  for (Layer* layer : layers)
    topological_sort_inner(layer, adjacent_list, &visited, &stack);
  layers.clear();
  while (!stack.empty()) {
    layers.push_back(stack.top());
    stack.pop();
  }
  std::reverse(layers.begin(), layers.end());
}

void runBackPropagation() {
  // sgd contains the hyper-parameters for stochastic gradient descent
  sgd = new SGD(model_conf_proto_.sgd());
  vector<Layer*> layers;
  vector<Edge*> edges;
  CreateNet(model_conf_proto_.layers(), &layers, &edges);
  // sort to make bottom layers be placed in the front positions
  // forward propagation is then based on this order
  topology_sort(layers);
  // reverse the orders for backward propagation
  vector<Layer*> reverse_layers(layers.rbegin(), layers.rend());

  vector<Param*> params;
  vector<Edge*> input_edges;
  for (auto* layer : layers) {
    // setup edges/parameters related to this layer
    layer.Setup();
    // prepare edges that accept input data
    for (auto *edge : layer.in_edges()) {
      for (auto& data_source : model_conf_proto_.data()) {
        if (data_source.name() == edge.name()) {
          input_edges.push_back(edge);
          continue;
        }
      }
    }
    // collect all parameters
    for (auto& param : layer.params())
      params.push_back(&param);
  }

  // TODO(Jingyang) fill all params from distributed memory
  // model_controller_.GetParam(params)

  while(!sgd->Finished()) {
    for (auto* layer : layers)
      for (auto * edge : layer.in_edges())
        // TODO(Jingyang) edge.name is the data source/table name,
        // fill edge.blob with mini-batch records
        // model_controller_.GetNextInput(edge.name(), edge.blob());
    // model_controller_.Get(&params);

    for (auto layer : layers)
      layer.Forward();

    for (auto layer : reverse_layers)
      layer.Backward();

    sgd->UpdateHyperParams();
    for (auto layer : layers) {
      layer.ComputeParamUpdates(sgd);
      // should be updated by model_controller as follows
      layer.UpdateParams();
    }

    sgd->IncStep();
    // TODO(Jingyang) update params in distributed memory
    // model_controller_.Update(&params);
  }
}

void runContrastiveDivergence() {
}

void Woker::WorkerMainFunc() {
  /*
  while(True) {
    recMsg

    switch msg


  }
  */
}

void Worker::run() {
  pthread_create(&app_thread, NULL, WorkerMainFunc, NULL);
}
}  // namespace lapis
