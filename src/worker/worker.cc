// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-04 11:49

#include "worker.h"

namespace lapis {
Worker::Worker(const DistributedMemory* dm,
               const DistributedDisk* dd,
               const GlobalContext* gc) {
  model_controller_ = new ModelController(dm, dd, gc);
}

Worker::vector<Layer *> CreateLayers(ModelConfProto& model_conf_proto_) {
  vector<Layer*> layers;
  for (LayerProto& layer_proto : model_conf_proto_.layers()) {
    Layer* layer = layer_factory_get(layer_proto.name());
    layer->init(layer_proto);
    layers.push_back(layer);
  }
  // fetch value of params
  model_controller_.FetchParam(layers);
  return layers;
}

void runBackPropagation() {
  vector<Layer> layers = CreateLayers(model_conf_proto_);
  vector<Layer> reverse_layers(layers.rbegin(), layers.rend());
  // layers = topology_sort(layers);  // sort forward order
  for (int i = 0; i < model_conf_proto_.num_batches(); i++) {
    for (auto layer: layers)
      if (layer.isInputLayer())
        model_controller_.GetNextInput(layer);

    for (auto layer: layers)
      layer.Forward();

    for (auto layer: reverse_layers)
      layer.Backward();

    model_controller_.Update(&layers);
  }
}

void runContrastiveDivergence() {

}

void Woker::WorkerMainFunc() {
  while(True) {
    recMsg

    switch msg


  }
}

void Worker::run() {
  pthread_create(&app_thread, NULL, WorkerMainFunc, NULL);

}
}  // namespace lapis
