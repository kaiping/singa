// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-04 11:14

#ifndef INCLUDE_WORKER_WORKER_H_
#define INCLUDE_WORKER_WORKER_H_

namespace lapis {
class Worker {
 public:
  Worker(const DistributedMemory* dm,
         const DistributedDisk* dd,
         const GlobalContext* gc);
  void run();

 private:
  ModelController model_controller_;
};
}  // namespace lapis


#endif  // INCLUDE_WORKER_WORKER_H_
