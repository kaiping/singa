#ifndef INCLUDE_UTILS_SGD_H_
#define INCLUDE_UTILS_SGD_H_
#include "proto/model.pb.h"
#include "utils/param.h"

namespace singa{
/**
 * base sgd class.
 */
class ParamUpdater{
 public:
  virtual void Init(const UpdaterProto &proto){
    proto_=proto;
  }
  virtual void Update(int step, Param* param)=0;

  float GetLearningRate(int step);
 protected:
  UpdaterProto proto_;
};

class AdaGradUpdater : public ParamUpdater{
 public:
  virtual void Init(const UpdaterProto& proto);
  virtual void Update(int step, Param* param);

 protected:
  float base_lr_;
  float delta_;
  float weight_decay_;
};

}

#endif // INCLUDE_UTILS_SGD_H_
