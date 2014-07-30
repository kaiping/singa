// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-15 11:11
#include <glog/logging.h>
#include "model/data_layer.h"

namespace lapis {

const std::string DataLayer::kType = "Data";

void DataLayer::Init(const LayerProto &proto) {
  Layer::Init(proto);
  data_source_name_ = proto.data_source();
  data_source_ = nullptr;
}

void DataLayer::ToProto(LayerProto *layer_proto) {
  Layer::ToProto(layer_proto);
  layer_proto->set_data_source(data_source_name_);
}

void DataLayer::Setup(int batchsize, TrainerProto::Algorithm alg,
                      const std::vector<DataSource *> &sources) {
  for (auto *source : sources) {
    if (source->name() == data_source_name_) {
      data_source_ = source;
      break;
    }
  }
  CHECK(data_source_ != nullptr) << "Cannot find data source for " << name_;
  data_.Resize(Shape4(data_source_->width(), data_source_->height(),
                      data_source_->channels(), batchsize);
}

void DataLayer::Forward() {
  data_source_->GetData(&data_);
}

void DataLayer::Backward() {
  for (Edge *edge : out_edges_) {
    Layer *layer = edge->OtherSide(this);
    edge->Backward(layer->feature(edge), layer->gradient(edge), data_,
                   nullptr, true);
  }
}
}  // namespace lapis
