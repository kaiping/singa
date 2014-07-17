// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-06 15:19

#include <glog/logging.h>
#include "model/layer.h"

namespace lapis {
/*****************************************************
 * Implementation for Layer
 *****************************************************/
void Layer::Init(const LayerProto &layer_proto,
                 const std::map<std::string, Edge *> &edge_map) {
  name_ = layer_proto.name();
  for (auto& edge_name : layer_proto.out_edge()) {
    CHECK(edge_map.find(edge_name)!= edge_map.end())
        << "No out going edge named '" << edge_name
        << "' for layer '" << name_ << "\n";
    Edge *edge = edge_map.at(edge_name);
    out_edges_.push_back(edge);
    if (edge->Bottom() == nullptr)
      edge->SetBottom(this);
  }

  for (std::string edge_name : layer_proto.in_edge()) {
    CHECK(edge_map.find(edge_name)!=edge_map.end())
        << "No incoming edge named '" << edge_name
        << "' for layer '" << name_ << "\n";
    Edge *edge = edge_map.at(edge_name);
    in_edges_.push_back(edge);
    if (edge->Top() == nullptr)
      edge->SetTop(this);
  }
}

void Layer::ToProto(LayerProto *proto) {
  proto->set_name(name_);
  for (Edge *edge : in_edges_)
    proto->add_in_edge(edge->Name());
  for (Edge *edge : out_edges_)
    proto->add_out_edge(edge->Name());
}

inline const std::string &Layer::Name() {
  return name_;
}

/**********************************************
 * Implementation for LayerFactory
 *********************************************/
LayerFactory* LayerFactory::Instance() {
  static LayerFactory factory;
  return &factory;
}

void LayerFactory::RegisterCreateFunction(
    const std::string id,
    std::function<Layer*(void)> create_function) {
  layer_map_[id]=create_function;
}

Layer* LayerFactory::Create(const std::string id) {
  Layer* instance = nullptr;

  auto it = layer_map_.find(id);
  if (it != layer_map_.end())
    instance = it->second();
  if (instance != nullptr)
    return instance;
  else
    return nullptr;
}

}  // namespace lapis
