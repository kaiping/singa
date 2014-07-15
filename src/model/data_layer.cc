// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-15 11:11

#include "model/data_layer.h"

namespace lapis {
void DataLayer::Init(const LayerProto &layer_proto,
                     const map<string, Edge *> &edge_map) {
  Layer::Init(layer_proto, edge_map);
  data_source_name_ = layer_proto.data_source();
  data_source_ = nullptr;
}

void DataLayer::ToProto(LayerProto *layer_proto) {
  Layer::ToProto(layer_proto);
  layer_proto.set_data_source(data_source_name_);
}

void DataLayer::Setup(int batchsize, Trainer::Algorithm alg,
                      const vector<DataSource *> &sources) {
  for (auto *source : sources) {
    if (source->name() == data_source_name_) {
      data_source_ = source;
      break;
    }
  }
  CHECK_NOTNULL(data_source_) << "Cannot find data source for layer '"
                              << name_ << "'\n";
  data_.Reshape(batchsize, data_source_.channels(), data_source_.height(),
                data_source_.width());
}

void DataLayer::Forward() {
  data_source_->GetData(&data_);
}

void DataLayer::Backward() {
  for (Edge *edge : out_edges_) {
    edge->Backward(edge->OtherSide()->Gradient(), nullptr);
  }
}

void DataLayer::ComputeParamUpdates(const Trainer *trainer) {
  LOG(INFO) << "ComputeParamUpdates() for datalyer does nothing\n";
}

inline bool DataLayer::HasInput() {
  return true;
}

inline Blob &Feature(Edge *edge) {
  return data_;
}

inline Blob &Gradient(Edge *edge) {
  return data_;
}
}  // namespace lapis
