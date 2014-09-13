// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-03 17:35

#ifndef INCLUDE_MODEL_PARAM_H_
#define INCLUDE_MODEL_PARAM_H_

#include <vector>
#include <string>
#include <map>
#include <functional>
#include <random>

#include "proto/model.pb.h"
#include "net/lapis.h"

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
  virtual void Init(const ParamProto &proto, const char flag);
  /**
   * Marshal properties, content and history gradient of this parameter into
   * ParamProto
   */
  virtual void ToProto(ParamProto *proto);
  /**
   * Return const mem address for the content of this parameter
   */
  const Blob &content() {
    return content_;
  }
  /**
   * Return mem address for the content of this parameter
   */
  Blob &mutable_content() {
    return content_;
  }
  /**
   * Return const mem address for the gradient of this parameter
   */
  const Blob &gradient() {
    return grad_;
  }
  /**
   * Return mem address for the gradient of this parameter
   */
  Blob &mutable_gradient() {
    return grad_;
  }
  /**
   * Return const mem address for the history gradient of this parameter
   */
  const Blob &history() {
    return history_grad_;
  }
  /**
   * Return mem address for the history gradient of this parameter
   */
  Blob &mutable_history() {
    return history_grad_;
  }
  /**
   * Return num of floats for this (vector) parameter
   */
  const int length() {
    return content_.length();
  }
  int id() {
    return id_;
  }
  void set_id(int id) {
    id_ = id;
  }
  std::string name() {
    return name_;
  }

  float momentum() {
    return momentum_;
  }
  float learning_rate() {
    return learning_rate_;
  }
  float weight_decay() {
    return weight_decay_;
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
  float momentum_, learning_rate_, weight_decay_;
  //! content, gradient and history gradient of this parameter
  Blob content_, grad_, history_grad_;
  /**
   * Currently support 5 init methods. May change to ParamInitFactory later to
   * support user defined init method.
   */
  ParamProto::InitMethod init_method;
};


/**
 * macro for register parameter init functions
 * @param TYPE the identifier of this init function
 * @param FUNC  the init function
 */
#define REGISTER_PARAM_INIT_FUNC(ID, FUNC) \
  ParamInitFactory::Instance()->RegisterInitFunc(ID, FUNC)
/**
 * Parameter initialization function factory.
 * It registers the user defined parameter initialization functions at runtime.
 * It also return this function when the function identifier is provided
 */
class ParamInitFactory {
 public:
  static ParamInitFactory *Instance();
  /**
   * Register the init function.
   * This method is called by the register macro REGISTER_PARAM_INIT_FUNC
   * @param id identifier the function, e.g, "Gaussian", i.e., the initializer
   * field in ParamProto
   * @param func std::function object
   */
  void RegisterInitFunc(std::string id,
                        const std::function<void(Param *)> &func);
  std::function<void(Param *)> &Get(std::string id);
 private:
  ParamInitFactory() {}
  std::map<std::string, std::function<void(Param *)>> map_;
};
}  // namespace lapis

#endif  // INCLUDE_MODEL_PARAM_H_
