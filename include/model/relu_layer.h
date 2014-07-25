// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-23 15:21
#ifndef INCLUDE_MODEL_RELU_LAYER_H_
#define INCLUDE_MODEL_RELU_LAYER_H_

#include "layer.h"
#include "proto/lapis.pb.h"

namespace lapis {
/**
 * Rectified linear unit layer.
 * The activation function is b=max(a,0), a is input value, b is output value.
 */
class ReLULayer : public Layer {
 public:
  static const std::string type;

  virtual void Setup(int batchsize, TrainAlgorithm alg,
                     const std::vector<DataSource *> &sources);
  virtual void Forward();
  virtual void Backward();
  virtual bool HasInput() {return false;}
  virtual Blob *feature(Edge *edge) {
    return edge->bottom()==this? &fea_:&act_;
  }
  virtual Blob *gradient(Edge *edge) {
    return edge->bottom()==this? &fea_grad_:&act_grad_;
  }

 private:
  //! fea short for feature, act short for activation, grad short for gradient
  Blob fea_, fea_grad_, act_, act_grad_;
};

}  // namespace lapis

#endif  // INCLUDE_MODEL_RELU_LAYER_H_
