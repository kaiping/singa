// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-23 16:20
#ifndef INCLUDE_MODEL_LINEAR_LAYER_H_
#define INCLUDE_MODEL_LINEAR_LAYER_H_

#include "model/layer.h"
namespace lapis {
/**
 * LinearLayer whose activation function is dummy.
 * This layer is used to connect edges.
 */
class LinearLayer : public Layer {
 public:
  /**
   * There are only two data blobs, one for feature, one for gradient.
   * This function allocate memory for then by calling the ::SetupTopBlob()
   * from the in coming edge.
   * @param batchsize not used
   * @param alg not used
   * @param sources not used
   */
  virtual void Setup(int batchsize, TrainerProto::Algorithm alg,
                     const std::vector<DataSource *> &sources);
  /**
   * Call ::Forward() function of in coming Edge.
   * Assume only one in coming edge currently.
   */
  virtual void Forward();
  /**
   * Call ::Backward() function of out going edge.
   * Assume only one out going edge currently.
   */
  virtual void Backward();
  virtual Blob *feature(Edge *edge) {return &fea_;}
  virtual Blob *gradient(Edge *edge) {return &grad_;}
 private:
  Blob fea_, grad_;
};

}  // namespace lapis

#endif  // INCLUDE_MODEL_LINEAR_LAYER_H_
