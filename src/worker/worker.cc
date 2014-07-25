// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-04 11:49

#include "worker/worker.h"

using std::vector;

namespace lapis {
Worker::Worker(const DistributedMemory *dm,
               const DistributedDisk *dd,
               const GlobalContext *gc) {
  model_controller_ = new ModelController(dm, dd, gc);
}

void runContrastiveDivergence() {
}

void Woker::WorkerMainFunc() {
  /*
  while(True) {
    recMsg

    switch msg


  }
  */
}

void Worker::run() {
  /*
   * do works
   */
  Finish();
}
}  // namespace lapis
