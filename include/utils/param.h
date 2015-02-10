#ifndef INCLUDE_MODEL_PARAM_H_
#define INCLUDE_MODEL_PARAM_H_

#include <vector>
#include <string>
#include <map>
#include <functional>
#include "proto/model.pb.h"
#include "utils/blob.h"
// Base paramter class.
namespace singa {
class Param {
 public:
  /**
   * Set properties of this parameter from ParamProto, allocate
   * corresponding memory and initialize the parameter. Copy data, history and
   * grad from ParamProto if available.
  void FromProto(const ParamProto &proto);
   */
  /**
   * Marshal properties, content and history gradient of this parameter into
   * ParamProto
  void ToProto(ParamProto *proto, bool copyData);
   */

   /**
    * @return num of floats.
    */
  int size() const {
    return data_.count();
  }
  /**
   * Return const mem address for the content of this parameter
   */
  const Blob<float> &data() {
    return data_;
  }
  Blob<float> *mutable_data() {
    return &data_;
  }
  /**
   * Return gradient of this parameter
   */
  const Blob<float> &grad() {
    return grad_;
  }
  Blob<float> *mutable_grad() {
    return &grad_;
  }

  const Blob<float> &history() {
    return history_;
  }
  Blob<float> *mutable_history() {
    return &history_;
  }

  float* mutable_cpu_data(){
    return data_.mutable_cpu_data();
  }
  const float* cpu_data(){
    return data_.cpu_data();
  }
  float* mutable_cpu_grad(){
    return grad_.mutable_cpu_data();
  }
  const float* cpu_grad(){
    return grad_.cpu_data();
  }
  void Setup(const ParamProto& proto, const std::vector<int>& shape);
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
 protected:
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
  Blob<float> data_, grad_, history_;
  /**
   * Currently support 5 init methods. May change to ParamInitFactory later to
   * support user defined init method.
   */
  ParamProto param_proto_;
};

}  // namespace singa

#endif  // INCLUDE_MODEL_PARAM_H_
