// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-14 20:46

#include <glog/logging.h>
#include "model/logistic_layer.h"
#include "utils/lapis.h"

namespace lapis {
const std::string LogisticLayer::type = "Logistic";

void LogisticLayer::Init(const LayerProto &proto) {
  Layer::Init(proto);
}

void LogisticLayer::Setup(int batchsize, TrainerProto::Algorithm alg,
                          const std::vector<DataSource *> &sources) {
  if (alg == TrainerProto::kBackPropagation) {
    Edge* edge=in_edges_[0];
    edge->SetupTopBlob(&activation_);
    edge->SetupTopBlob(&feature_);
    edge->SetupTopBlob(&activation_grad_);
    edge->SetupTopBlob(&feature_grad_);
    feature_dimension_=activation_.record_length();
  } else {
    //! CD or PCD
    /*
    pos_feature_.Reshape(batchsize, feature_dimension_);
    neg_feature_.Reshape(batchsize, feature_dimension_);
    */
  }
}

void LogisticLayer::ToProto(LayerProto *proto) {
  Layer::ToProto(proto);
  proto->set_type(type);
}

void LogisticLayer::Forward() {
  CHECK_GE(in_edges_.size(),
           1) << "logistic layer must have >=1 incoming edges";
  auto it = in_edges_.begin();
  Edge *edge = (*it);
  edge->Forward(edge->OtherSide(this)->feature(edge), &activation_, true);
  for (it++ ; it != in_edges_.end(); it++) {
    edge = *it;
    edge->Forward(edge->OtherSide(this)->feature(edge), &activation_, false);
  }
  AVec activation(activation_.mutable_data(), feature_dimension_);
  AVec feature(feature_.mutable_data(), feature_dimension_);
  feature = 1./(1.+(-activation).exp());
}

void LogisticLayer::Backward() {
  CHECK_GE(out_edges_.size(), 1) << "logistic layer must have >=1 out edges";
  auto it = out_edges_.begin();
  Edge *edge = *it;
  Layer *layer=edge->OtherSide(this);
  edge->Backward(layer->feature(edge), layer->gradient(edge), &feature_,
                 &feature_grad_, true);
  for (it++; it != out_edges_.end(); it++) {
    edge = *it;
    layer=edge->OtherSide(this);
    edge->Backward(layer->feature(edge), layer->gradient(edge), &feature_,
                   &feature_grad_, false);
  }
  AVec feature_grad(feature_grad_.mutable_data(),feature_dimension_);
  AVec feature(feature_.mutable_data(), feature_dimension_);
  AVec activation_grad(activation_grad_.mutable_data(),feature_dimension_);
  activation_grad = feature_grad * feature * (1 - feature);
}
}  // namespace lapis
