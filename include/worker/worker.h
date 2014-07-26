// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-04 11:14

#ifndef INCLUDE_WORKER_WORKER_H_
#define INCLUDE_WORKER_WORKER_H_
#include "model_controller/model.h"
#include "utils/global_context.h"


namespace lapis {
class Worker {
 public:
  Worker(ModelController *mc);
  void Run();

 private:
  GlobalContext *global_context_;
  ModelController *model_controller_;
};
}  // namespace lapis

#endif  // INCLUDE_WORKER_WORKER_H_
