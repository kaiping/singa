// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-23 22:02

#ifndef INCLUDE_MODEL_LRN_EDGE_H_
#define INCLUDE_MODEL_LRN_EDGE_H_
#include "proto/model.pb.h"
#include "model/edge.h"
namespace lapis {
/**
 * Local Response Normalization edge
 * b_i=a_i/x_i^beta
 * x_i=k+alpha*\sum_{j=max(0,i-n/2}^{min(N,i+n/2}(a_j)^2
 * n is size of local response area.
 * a_i, the activation (after ReLU) of a neuron convolved with the i-th kernel.
 * b_i, the neuron after normalization, N is the total num of kernels
 */
class LRNEdge : public Edge {
 public:
  virtual void Init(const EdgeProto &proto,
                    const std::map<std::string, Layer *> &layer_map);
  virtual void Setup(bool set_param);
  virtual void ToProto(EdgeProto *proto);
  virtual void Forward(const Blob4 &src, Blob4 *dest, bool overwrite);
  virtual void Backward(const Blob4 &src_fea, const Blob4 &src_grad,
                        const Blob4 &dest_fea, Blob4 *dest_grad,
                        bool overwrite);

  virtual void SetupTopBlob(Blob4 *blob);

 private:
  //! shape of the bottom layer feature
  int num_, channels_, height_, width_;
  //! size local response (neighbor) area and padding size
  int local_size_, pre_pad_;
  //! hyper-parameter
  float alpha_, beta_;
  //! accumulate local neighbor feature, i.e., the x_i;
  Blob3 accum_fea_;
  //! tmp grad used in accumulating the gradient in BackProp
  Blob1 accum_grad_;
  //! the square of feature with padding area
  Blob2 pad_square_;
  //! gradient of feature with padding area
  Blob2 pad_grad_;
};

}  // namespace lapis

#endif  // INCLUDE_MODEL_LRN_EDGE_H_
