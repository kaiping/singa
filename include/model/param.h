// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-03 17:35

#ifndef INCLUDE_MODEL_PARAM_H_
#define INCLUDE_MODEL_PARAM_H_

#include <vector>
#include <string>
#include "proto/lapis.pb.h"
#include "model/blob.h"

// Base paramter class.
// TODO(Jingyang) define split/partition function.
namespace lapis {
class Param {
 public:
  /**
   * Set properties of this parameter from ParamProto, allocate
   * corresponding memory and initialize the parameter
   */
  virtual void Init(const ParamProto &param_proto);
  //! Marshal properties of this parameter into google protobuf
  virtual void ToProto(ParamProto *param_proto);
  //! Return data pointer for this parameter
  float *Content() {
    return content_.Content();
  }
  //! Return num of rows for matrix parameters
  const int Rows() {
    return content_.Height();
  }
  //! Return num of columns for matrix parameters
  const int Cols() {
    return content_.Width();
  }
  //! Return num of floats for vector parameters
  const int Length() {
    return content_.Size();
  }

 protected:
  Blob content_, grad_, history_grad_;
  float learning_rate_, weight_decay_;
  std::string initializer_;
  std::string name_;  //!< name of the parameter, e.g., 'weight', 'bias'
};
}  // namespace lapis

#endif  // INCLUDE_MODEL_PARAM_H_
