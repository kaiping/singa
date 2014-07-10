// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-04 11:49

#include "worker/worker.h"

namespace lapis {
Worker::Worker(const DistributedMemory* dm,
               const DistributedDisk* dd,
               const GlobalContext* gc) {
  model_controller_ = new ModelController(dm, dd, gc);
}

Worker::vector<Layer *>& CreateLayers(const ModelConfProto& model_conf_proto_) {
  vector<Layer*> *layers = new vector<Layer*>();
  for (LayerProto& layer_proto : model_conf_proto_.layers()) {
    Layer* layer = layer_factory_get(layer_proto.name());
    layer->init(layer_proto);
    layers->push_back(layer);
  }
  return *layers;
}

bool HasIntersection(const std::vector<Blob*>& A, const std::vector<Blob*>& B) {
  for (Blob* a : A)
    for (Blob* b : B)
      if (a == b)
        return true;
  return false;
}

void topological_sort(Layer* layer,
                      const std::map<Layer*, vector<Layer*>>& adjacent_list,
                      std::map<Layer*, bool>* visited,
                      std::stack<Layer*>* stack) {
  (*visited)[layer] = true;
  for (Layer* layer1 : adjacent_list[layer]) {
    if ((*visited)[layer])
      continue;
    topological_sort(layer1, adjacent_list, visited, stack);
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
    topological_sort(layer, adjacent_list, &visited, &stack);
  layers.clear();
  while (!stack.empty()) {
    layers.push_back(stack.top());
    stack.pop();
  }
  std::reverse(layers.begin(), layers.end());
}

void runBackPropagation() {
  vector<Layer*> layers = CreateLayers(model_conf_proto_);
  topology_sort(layers);  // sort forward order
  vector<Layer*> reverse_layers(layers.rbegin(), layers.rend());

  vector<Param*> params;
  for (auto* layer : layers) {
    for (auto *edge : layer.in_edges()) {
      if (edge.data() == nullptr) {
        for (auto& data_source : model_conf_proto_.data()) {
          if (data_source.name() == edge.name()) {
            edge.GetInput(data_source);
            continue;
          }
        }
      }
    }
    // setup blobs of out edges
    layer.setup();
    for (auto& param : layer.params())
      params.push_back(&param);
  }

  // TODO(Jingyang) fill all params from distributed memory
  model_controller_.GetParam(params)

  for (int i = 0; i < model_conf_proto_.num_batches(); i++) {
    for (auto* layer : layers)
      for (auto * edge : layer.in_edges())
        // TODO(Jingyang) edge.name is the data source/table name,
        // fill edge.blob with mini-batch records
        model_controller_.GetNextInput(edge.name(), edge.blob());

    for (auto layer : layers)
      layer.Forward();

    for (auto layer : reverse_layers)
      layer.Backward();

    // TODO(Jingyang) update params in distributed memory
    model_controller_.Update(&params);
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
