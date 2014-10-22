// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-03 17:35

#ifndef INCLUDE_NET_PARAM_H_
#define INCLUDE_NET_PARAM_H_

#include <vector>
#include <string>
#include <map>
#include <functional>
#include "da/dary.h"
#include "proto/model.pb.h"

// Base paramter class.
// TODO(Jingyang) define split/partition function.
namespace lapis {
class Param {
 public:
  /**
   * Set properties of this parameter from ParamProto, allocate
   * corresponding memory and initialize the parameter. Copy content and
   * history from ParamProto if available.
   */
  void Init(const ParamProto &proto);
  /**
   * Marshal properties, content and history gradient of this parameter into
   * ParamProto
   */
  void ToProto(ParamProto *proto, bool copyData);
  /**
   * Return const mem address for the content of this parameter
   */
  const DAry &data() {
    return data_;
  }
  /**
   * Return mem address for the content of this parameter
   */
  DAry *mutable_data() {
    return &data_;
  }
  /**
   * Return const mem address for the gradient of this parameter
   */
  const DAry &grad() {
    return grad_;
  }
  /**
   * Return mem address for the gradient of this parameter
   */
  DAry *mutable_grad() {
    return &grad_;
  }

  void SetShape(int h, int w);
  void SetShape(int l);
  void SetPartition(int k);
  void SetupDAry(int k);
  /*
   * fill the data according to initmethod, i.e., random/gaussian/fixed value
   */
  void Fill();

  int id() {
    return id_;
  }
  void set_id(int id) {
    id_ = id;
  }
  float learning_rate_multiplier() {
    return learning_rate_multiplier_;
  }
  float weight_decay_multiplier() {
    return weight_decay_multiplier_;
  }
  const int split_threshold(){
    return split_threshold_;
  }
 protected:
  /**
   * Fill in the val with data generated from a uniform distribution
   * @param low the lower boundary of the uniform distribution
   * @param high the upper boundary of the uniform distribution
   * @param factor the generated data is multiplied to this number
   * @param val float array to store the generated data
   */
  void FillUniformData(float low, float high, float factor);
  /**
   * Similar to ::FillGaussainData(), except the data are generated from
   * Gaussain distribution.
   */
  void FillGaussainData(float mean, float std, float factor);

  /**
   * name of the parameter used to identify the ParamProto configed in
   * EdgeProto by users. Currently there are two kinds of parameters, 'weight'
   * and 'bias'.
   */
  std::string name_;
  /**
   * identifier of this parameter, will be used by ModelController
   */
  int id_;
  //! scale factor for learning rate and weight decay for this parameter
  float momentum_, learning_rate_multiplier_, weight_decay_multiplier_;
  float low_, high_, mean_, std_, value_;
  int split_threshold_;
  //! content, gradient and history gradient of this parameter
  DAry data_, grad_;
  /**
   * Currently support 5 init methods. May change to ParamInitFactory later to
   * support user defined init method.
   */
  ParamProto::InitMethod init_method_;
};

}  // namespace lapis

#endif  // INCLUDE_NET_PARAM_H_
