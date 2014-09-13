// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-09-12 12:39

#ifndef INCLUDE_NET_PREDICTION_LAYER_H
#define INCLUDE_NET_PREDICTION_LAYER_H
#include "proto/model.pb.h"
#include "net/data_layer.h"

namespace lapis {
class PredictionLayer : public DataLayer {
 public:
  virtual Performance CalcPerf(bool loss=true, bool precision=true)=0;

 protected:
  Blob prediction_;
};

class SoftmaxPredictionLayer : public PredictionLayer {
 public:
  virtual void Setup(const char flag);
  virtual void Forward();
  virtual Performance CalcPerf(bool loss=true, bool precision=true);
  virtual Blob& gradient(Edge* edge) {
    return data_;
  }
  virtual Blob& feature(Edge* edge) {
    if(edge==nullptr)
      return data_;
    else
      return prediction_;
  }

 private:
  // if ground truth label is among the topk_ results,
  // then the prediction is correct
  int topk_;
};
}  // namespace lapis
#endif  // INCLUDE_NET_PREDICTION_LAYER_H
