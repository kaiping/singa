// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-14 20:46

#include <glog/logging.h>
#include "model/logistic_layer.h"
#include "utils/lapis.h"

namespace lapis {
const std::string kLogisticLayer = "Logistic";
void LogisticLayer::Init(const LayerProto &layer_proto,
                         const std::map<std::string, Edge *> &edge_map) {
  feature_dimension_ = layer_proto.num_output();
}


void LogisticLayer::Setup(int batchsize, TrainAlgorithm alg,
                          const std::vector<DataSource *> &sources) {
  if (alg == TrainAlgorithm::kBackPropagation) {
    activation_.Reshape(batchsize, feature_dimension_);
    feature_.Reshape(batchsize, feature_dimension_);
    activation_grad_.Reshape(batchsize, feature_dimension_);
    feature_grad_.Reshape(batchsize, feature_dimension_);
  } else {
    //! CD or PCD
    pos_feature_.Reshape(batchsize, feature_dimension_);
    neg_feature_.Reshape(batchsize, feature_dimension_);
  }
}

void LogisticLayer::ToProto(LayerProto *layer_proto) {
  layer_proto->set_name(name_);
  layer_proto->set_num_output(feature_dimension_);
  layer_proto->set_type(kLogisticLayer);
}

void LogisticLayer::Forward() {
  MapArrayType act(activation_.MutableContent(), activation_.Height(),
                   activation_.Width());
  CHECK_GE(in_edges_.size(),
           1) << "logistic layer must have >=1 incoming edges\n";

  auto it = in_edges_.begin();
  Edge *edge = (*it);
  edge->Forward(edge->OtherSide(this)->Feature(edge), &activation_, true);
  for (it++ ; it != in_edges_.end(); it++) {
    edge = *it;
    edge->Forward(edge->OtherSide(this)->Feature(edge), &activation_, false);
  }

  MapArrayType fea(feature_.MutableContent(), feature_.Height(),
                   feature_.Width());
  fea = (-act).exp();
  fea = 1. / (1. + fea);
}

void LogisticLayer::Backward() {
  CHECK_GE(out_edges_.size(), 1) << "logistic layer must have >=1 out edges\n";

  auto it = out_edges_.begin();
  Edge *edge = *it;
  edge->Backward(edge->OtherSide(this)->Gradient(edge), &feature_,
                 &feature_grad_, true);
  for (it++; it != out_edges_.end(); it++) {
    edge = *it;
    edge->Backward(edge->OtherSide(this)->Gradient(edge), &feature_,
                   &feature_grad_, false);
  }
  MapArrayType act_grad(activation_grad_.MutableContent(),
                        activation_grad_.Height(),
                        activation_grad_.Width());
  MapArrayType fea_grad(feature_grad_.MutableContent(), feature_grad_.Height(),
                        feature_grad_.Width());
  MapArrayType fea(feature_.MutableContent(), feature_.Height(),
                   feature_.Width());
  act_grad = fea_grad * fea * (1 - fea);
}


}  // namespace lapis
