// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-06 15:19

#include "worker/layer.h"

namespace lapis {
void Layer::Init(const LayerProto &layer_proto,
                 const map<string, Edge *> &edge_map) {
  name_ = layer_proto.name();
  for (string &edge_name : layer_proto.out_edges()) {
    CHECK_NE(edge_map.find(edge_name), edge_map::end())
        << "No out going edge named '" << edge_name
        << "' for layer '" << name_ << "\n";
    Edge *edge = edge_map[edge_name];
    out_edges_.push_back(edge);
    if (edge->Bottom() == nullptr)
      edge.SetBottom(this);
  }

  for (string &edge_name : layer_proto.in_edges()) {
    CHECK_NE(edge_map.find(edge_name), edge_map::end())
        << "No incoming edge named '" << edge_name
        << "' for layer '" << name_ << "\n";
    Edge *edge = edge_map[edge_name];
    in_edges_.push_back(edge);
    if (edge->Top() == nullptr)
      edge.SetTop(this);
  }
}

void Layer::ToProto(LayerProto *proto) {
  proto->set_name(name_);
  for (Edge *edge : in_edges_)
    proto->add_in_edge(edge->Name());
  for (Edge *edge : out_edges_)
    proto->add_out_edge(edge->Name());
}

inline const string &Layer::Name() {
  return name_;
}

}  // namespace lapis
