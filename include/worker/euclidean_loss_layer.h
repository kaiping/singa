// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-11 10:56

#ifndef INCLUDE_WORKER_EUCLIDEAN_LOSS_LAYER_H_
#define INCLUDE_WORKER_EUCLIDEAN_LOSS_LAYER_H_
namespace lapis {
class EuclideanLossLayer : public Layer
{
 public:
  virtual void Setup(const LayerProto& layer_proto, std::map<string, Edge*>* edges);

  virtual void Forward();
  virtual void Backward();

 private:
  MapMatrixType label_, predict_, grad_;
  float loss_;
};

}  // namespace lapis
#endif  // INCLUDE_WORKER_EUCLIDEAN_LOSS_LAYER_H_

