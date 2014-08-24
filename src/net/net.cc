// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-14 13:18

#include <map>
#include <stack>
#include "net/data_layer.h"
#include "net/net.h"

namespace lapis {
// visited all out going layers and then push current layer into the stack
void topology_sort_inner(Layer *layer,
                         const std::map<Layer *,
                         std::vector<Layer *>> &adjacent_list,
                         std::map<Layer *, bool> *visited,
                         std::stack<Layer *> *stack) {
  (*visited)[layer] = true;
  for (Layer *layer1 : adjacent_list.at(layer)) {
    if ((*visited)[layer1])
      continue;
    topology_sort_inner(layer1, adjacent_list, visited, stack);
  }
  stack->push(layer);
}

// sort to make `bottom' layers be placed in the front positions
// forward propagation will be processed based on this order
void topology_sort(std::vector<Layer *> *layers) {
  // adjacent list from upper layers to lower layers
  std::map<Layer *, std::vector<Layer *>> adjacent_list;
  std::map<Layer *, bool> visited;
  std::vector<Layer *> input_layers;
  // prepare adjacent list; input layers will be processed firstly,
  // hence no need to sort them (mark them as visited)
  for (Layer *layer : *layers) {
    /*
    if (layer->HasInput()) {
      visited[layer] = true;
      input_layers.push_back(layer);
    } else {
      visited[layer] = false;
    }
    */
    // automatically insert a new entry to the map
    // the direction of edge is from layer to layers in adjacent_list
    adjacent_list[layer];
    for (Edge *edge : layer->out_edges()) {
      Layer *layer1 = edge->OtherSide(layer);
      adjacent_list[layer].push_back(layer1);
    }
  }
  // the `top' layer in the net will be placed at the bottom of the stack
  // and then be processed (i.e., forward) at last
  std::stack<Layer *> stack;
  for (Layer *layer : *layers) {
    if (visited[layer] == false)
      topology_sort_inner(layer, adjacent_list, &visited, &stack);
  }
  layers->clear();
  // input layers are placed at front to be processed firstly
  for (auto layer : input_layers)
    layers->push_back(layer);
  while (!stack.empty()) {
    layers->push_back(stack.top());
    stack.pop();
  }
}

Net::Net(const NetProto &net_proto) {
  LOG(INFO)<<"Init Neural Net";
  std::map<std::string, Layer *> layer_map;
  for (auto &layer_proto : net_proto.layer()) {
    Layer *layer = LayerFactory::Instance()->Create(layer_proto.type());
    layer->Init(layer_proto);
    layers_.push_back(layer);
    layer_map[layer->name()] = layer;
  }
  int param_id = 0;
  for (auto &edge_proto : net_proto.edge()) {
    Edge *edge = EdgeFactory::Instance()->Create(edge_proto.type());
    edge->Init(edge_proto, layer_map);
    edges_.push_back(edge);
    for (auto *param : edge->params()) {
      param->set_id(param_id++);
      params_.push_back(param);
    }
  }
  topology_sort(&layers_);
  LOG(INFO)<<"Neural Net constructed";
}
void Net::Setup(const char flag,const int batchsize,
                const std::map<std::string, Shape> &shapes){
  VLOG(3)<<"Setup net allocate data: "<<AllocData(flag);
  for (auto *layer : layers()){
    if (layer->HasInput()){
      VLOG(3)<<layer->name();
      DataLayer* dlayer=dynamic_cast<DataLayer*>(layer);
      std::string source=dlayer->data_source();
      CHECK(shapes.find(source)!=shapes.end());
      dlayer->SetInputShape(batchsize, shapes.at(source));
    }
  }
  for(auto *layer : layers()) {
    VLOG(3)<<layer->name();
    layer->Setup(flag);
    for (auto *edge : layer->out_edges()){
      VLOG(3)<<edge->name();
      edge->Setup(flag);
    }
  }
}

void Net::Setup(const char flag,const int batchsize,
                const std::map<std::string, Shape> &shapes,
                const std::map<std::string, int>& stores){
  Setup(flag, batchsize, shapes);
  for (auto *layer : layers()){
    if (layer->HasInput()){
      DataLayer* dlayer=dynamic_cast<DataLayer*>(layer);
      std::string source=dlayer->data_source();
      CHECK(shapes.find(source)!=shapes.end());
      dlayer->SetInputStore(stores.at(source));
    }
  }
}
void Net::ToProto(NetProto *net_proto) {
  for (Layer *layer : layers_) {
    LayerProto *layer_proto = net_proto->add_layer();
    layer->ToProto(layer_proto);
  }
  for (Edge *edge : edges_) {
    EdgeProto *edge_proto = net_proto->add_edge();
    edge->ToProto(edge_proto);
  }
}
Net::~Net() {
  for(auto* layer: layers_)
    delete layer;
  for(auto* edge: edges_)
    delete edge;
}
}  // namespace lapis
