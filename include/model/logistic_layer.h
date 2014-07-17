// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-14 19:40

#ifndef INCLUDE_MODEL_LOGISTIC_LAYER_H_
#define INCLUDE_MODEL_LOGISTIC_LAYER_H_
#include <map>
#include <string>
#include <vector>
#include "proto/lapis.pb.h"
#include "model/layer.h"
#include "model/trainer.h"

namespace lapis {
/**
 * Traditional neural network layer.
 * This layer is used in both RBM and feed-forward neural networks.
 */
class LogisticLayer : public Layer {
 public:
  /**
   * identifier for logistic layer
   * LayerFactory will create an instance of this layer based on this
   * identifier
   */
  static const std::string kLogisticLayer;

  virtual void Init(const LayerProto &layer_proto,
                    const std::map<std::string, Edge *> &edge_map);
  virtual void Setup(int batchsize, TrainAlgorithm alg,
                     const std::vector<DataSource *> &sources) = 0;
  /**
   * There may be multiple incoming edges, it sums data/activations from all
   * incoming edges as the activation and then applies the logistic function.
   */
  virtual void Forward();
  /**
   * There may be multiple outgoing edges, it sums gradients from all outgoing
   * edges, and then comptue the gradient w.r.t the aggregated activation
   */
  virtual void Backward();
  virtual void ComputeParamUpdates(const Trainer *trainer);
  virtual void ToProto(LayerProto *layer_proto);
  virtual bool HasInput() {
    return false;
  }
  virtual Blob *Feature(Edge *edge) {
    // TODO(wangwei) return pos_feature_ or neg_feature_ for PCD/CD algorithm
    return &feature_;
  }

  virtual Blob *Gradient(Edge *edge) {
    return &activation_grad_;
  }

 private:
  //! used in backpropagation method
  Blob activation_, feature_, activation_grad_, feature_grad_;
  //! used in (persistent) contrastive divergence method
  Blob pos_feature_, neg_feature_;
  //! dimension of this layer/feature
  int feature_dimension_;
};
}  // namespace lapis
#endif  // INCLUDE_MODEL_LOGISTIC_LAYER_H_
