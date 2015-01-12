#ifndef INCLUDE_NET_PARAM_H_
#define INCLUDE_NET_PARAM_H_

#include <vector>
#include <string>
#include <map>
#include <functional>
#include "da/darray.h"
#include "proto/model.pb.h"
using std::vector;
using std::string;
// Base paramter class.
namespace singa {
class Param {
 public:
  /**
   * Set properties of this parameter from ParamProto, allocate
   * corresponding memory and initialize the parameter. Copy data, history and
   * grad from ParamProto if available.
   */
  void FromProto(const ParamProto &proto);
  /**
   * Marshal properties, content and history gradient of this parameter into
   * ParamProto
   */
  void ToProto(ParamProto *proto, bool copyData);
  /**
   * Return const mem address for the content of this parameter
   */
  const DArray &data() {
    return data_;
  }
  DArray *mutable_data() {
    return &data_;
  }
  /**
   * Return gradient of this parameter
   */
  const DArray &grad() {
    return grad_;
  }
  /**
   * Return gradient history of this parameter
   */
  const DArray &history() {
    return history_;
  }
  DArray *mutable_grad() {
    return &grad_;
  }
  /**
   * Return mem address for the content of this parameter
   */
  DArray *mutable_history() {
    return &history_;
  }
  float* mutable_dptr(){
    return data_.dptr();
  }
  const float* dptr(){
    return data_.dptr();
  }
  const float* gptr(){
    return grad_.dptr();
  }
  float* mutable_gptr(){
    return grad_.dptr();
  }

  const int local_size(){
    return data_.local_size();
  }
  void Setup(const vector<int>& shape, int partition_dim);
  /*
   * fill the data according to initmethod, i.e., random/gaussian/fixed value
   */
  void Init();

  const std::string& name() {
    return param_proto_.name();
  }

  int id() {
    return param_proto_.id();
  }
  void set_id(int id) {
    param_proto_.set_id(id);
  }
  float learning_rate_multiplier() {
    return param_proto_.learning_rate_multiplier();
  }
  float weight_decay_multiplier() {
    return param_proto_.weight_decay_multiplier();
  }
  const int split_threshold(){
    return param_proto_.split_threshold();
  }
  const int partition(){
    return param_proto_.partition_dim();
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
  //! content, gradient and history gradient of this parameter
  DArray data_, grad_, history_;
  /**
   * Currently support 5 init methods. May change to ParamInitFactory later to
   * support user defined init method.
   */
  ParamProto param_proto_;
};

}  // namespace lapis

#endif  // INCLUDE_NET_PARAM_H_
