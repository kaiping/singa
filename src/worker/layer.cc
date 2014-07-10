// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-06 15:19

#include "worker/layer.h"

namespace lapis {
Layer::init(const LayerProto& layer_proto, std::map<string, Edge*>* edges) {
  for (ParamProto param_proto : layer_proto.params()) {
    Param* param = Param_factory_get(param_proto.type());
    param.init(param_proto);
    params_.push_back(param);
  }

  Edge* edge = nullptr;
  for (EdgeProto edge_proto : layer_proto.out_edges()) {
    if (edges.find(edge_proto.name()) == edges.end()) {
      edge = new Edge(edge_proto);
      (*edges)[edge_proto.name()] = edge;
    } else {
      edge = (*edges)[edge_proto.name()];
    }
    out_edges_.push_back(edge);
  }

  for (EdgeProto edge_proto : layer_proto.in_edges()) {
    if (edges.find(edge_proto.name()) == edges.end()) {
      edge = new Edge(edge_proto);
      (*edges)[edge_proto.name()] = edge;
    } else {
      edge = (*edges)[edge_proto.name()];
    }
    in_edges_.push_back(edge);
  }
}
}  // namespace lapis
