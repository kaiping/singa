// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-15 11:11
#include <glog/logging.h>
#include "model/data_layer.h"

namespace lapis {

const std::string kDataLayer = "Data";
void DataLayer::Init(const LayerProto &layer_proto,
                     const std::map<std::string, Edge *> &edge_map) {
  Layer::Init(layer_proto, edge_map);
  data_source_name_ = layer_proto.data_source();
  data_source_ = nullptr;
}

void DataLayer::ToProto(LayerProto *layer_proto) {
  Layer::ToProto(layer_proto);
  layer_proto->set_data_source(data_source_name_);
}

void DataLayer::Setup(int batchsize, TrainAlgorithm alg,
                      const std::vector<DataSource *> &sources) {
  for (auto *source : sources) {
    if (source->Name() == data_source_name_) {
      data_source_ = source;
      break;
    }
  }
  CHECK(data_source_ != nullptr) << "Cannot find data source for layer '" << name_
                                 << "'\n";
  data_.Reshape(batchsize, data_source_->Channels(), data_source_->Height(),
                data_source_->Width());
}

void DataLayer::Forward() {
  data_source_->GetData(&data_);
}

void DataLayer::Backward() {
  for (Edge *edge : out_edges_) {
    edge->Backward(edge->OtherSide(this)->Gradient(edge), &data_, nullptr, true);
  }
}

void DataLayer::ComputeParamUpdates(const Trainer *trainer) {
  LOG(INFO) << "ComputeParamUpdates() for datalyer does nothing\n";
}

}  // namespace lapis
