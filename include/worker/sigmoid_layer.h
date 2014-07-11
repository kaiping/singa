// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-10 22:46

#ifndef INCLUDE_WORKER_SIGMOID_LAYER_H_
#define INCLUDE_WORKER_SIGMOID_LAYER_H_

#include "worker/layer.h"

namespace lapis {
class SigmoidLayer : public Layer {
 public:
  virtual void init(const LayerProto& layer_proto,
                    std::map<string, Edge*>* edges);

  virtual void Forward() = 0;
  virtual void Backward() = 0;

 private:
  // operations for sigmoid layer are element-wise, hence must use Eigen::Array
  MapArrayType in_, in_grad_, out_, out_grad_;
}

}  // namespace lapis
#endif  // INCLUDE_WORKER_SIGMOID_LAYER_H_

