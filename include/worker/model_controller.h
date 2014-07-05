// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-04 19:58

#ifndef INCLUDE_WORKER_MODEL_CONTROLLER_H_
#define INCLUDE_WORKER_MODEL_CONTROLLER_H_

namespace lapis {
class ModelController {
 public:
  void GetNextInput(Layer *layer);
  void Update(vector<Layer> *layer);

 private:
  ModelConfProto model_conf_proto_;


};

}  // namespace lapis
#endif  // INCLUDE_WORKER_MODEL_CONTROLLER_H_

