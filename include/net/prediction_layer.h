// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-09-12 12:39

#ifndef INCLUDE_NET_PREDICTION_LAYER_H
#define INCLUDE_NET_PREDICTION_LAYER_H

namespace lapis {
class PredictionLayer : public DataLayer {
 public:
  virtual Performance CalcAccuracy()=0;

 private:
  Blob prediction_;
};


class SoftmaxPredictionLayer : PredictionLayer {
 public:
  virtual void Forward();
  virtual void Performance CalcAccuracy();
  virtual Blob& gradient(Edge* edge) {
    return data_;
  }
  virtual Blob& feature(Edge* edge) {
    return prediction_;
  }

 private:
  // if ground truth label is among the topk_ results,
  // then the prediction is correct
  int topk_;
};
}  // namespace lapis
#endif  // INCLUDE_NET_PREDICTION_LAYER_H
