// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-14 20:46

#include <glog/loggin.h>
#include "model/logistic_layer.h"

namespace lapis {
void LogisticLayer::Init(const LayerProto &layer_proto,
                         const map<string, Edge *> &edge_map) {
  feature_dimension_ = layer_proto.num_output();
}


void LogisticLayer::Setup(int batchsize, Trainer::Algorithm alg,
                          const vector<DataSource *> &sources);
if (alg == Trainer::Algorithm::kBackPropagation) {
  activation_.Reshape(batchsize, feature_dimension_);
  feature_.Reshape(batchsize, feature_dimension_);
  activation_grad_.Reshape(batchsize, feature_dimension_);
  feature_grad_.Reshape(batchsize, feature_dimension_);
} else {
  //! CD or PCD
  pos_feature_.Reshape(batchsize, feature_dimension_);
  neg_feature_.Reshape(batchsize, feature_dimension_);
}

void LogisticLayer::ToProto(LayerProto *layer_proto) {
  layer_proto->set_name(name_);
  layer_proto->set_num_output(num_output_);
  layer_proto->set_type(kLogistic);
}

void LogisticLayer::Forward() {
  MapArrayType act(activation_.Content(), activation_.Rows(),
                   activation_.Cols());
  CHECK_GE(in_edges_.size(),
           1) << "logistic layer must have >=1 incoming edges\n";
  Edge *edge = in_edges_[0];
  edge.Forward(act, true);
  for (int i = 1; i < in_edges_.size(); i++) {
    edge = in_edges_[i];
    edge->Forward(edge->OtherSide(this)->Feature(), activation_, false);
  }

  MapArrayType fea(feature_.Content(), feature_.Rows(), feature_.Cols());
  fea = (-act).exp();
  fea = 1. / (1. + fea);
}

void LogisticLayer::Backward() {
  CHECK_GE(out_edges_.size(), 1) << "logistic layer must have >=1 out edges\n";
  Edge *edge = out_edge_[0];
  edge->Backward(edge->OtherSide(this)->Gradient(), feature_, feature_grad_,
                 true);
  for (int i = 1; i < out_edges_.size(); i++) {
    edge = out_edges_[i];
    edge->Backward(edge->OtherSide(this)->Gradient(), feature_, feature_grad_,
                   false);
  }
  MapArrayType act_grad(activation_grad_.Content(), activation_grad_.Rows(),
                        activation_grad_.Cols());
  MapArrayType fea_grad(feature_grad_.Content(), feature_grad_.Rows(),
                        feature_grad_.Cols());
  MapArrayType fea(feature_.Content(), feature_.Rows(), feature_.Cols());
  act_grad.noalias() = fea_grad * fea * (1 - fea);
}

inline bool HasInput() {
  return false;
}

inline Blob &Feature(Edge *edge) {
  // TODO(wangwei) return pos_feature_ or neg_feature_ for PCD/CD algorithm
  return feature_;
}

inline Blob &Gradient(Edge *edge) {
  return activation_grad_;
}
}  // namespace lapis
