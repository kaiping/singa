#include "core/common.h"
#include "core/rpc.h"
#include "core/distributed-memory.h"
#include "core/memory-server.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <signal.h>
#include <execinfo.h>
#include <fcntl.h>

#include <math.h>

//#include <asm/msr.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <sys/types.h>

//#include <lzo/lzo1x.h>
#include <mpi.h>

DEFINE_bool(localtest, false, "");
DEFINE_bool(run_tests, false, "");

DEFINE_string(hostfile, "conf/mpi-beakers", "");
DEFINE_int32(workers, 2, "");

namespace lapis {

//DistributedMemoryManager *manager;
//MemoryServer *server;

void Sleep(double t) {
  timespec req;
  req.tv_sec = (int)t;
  req.tv_nsec = (int64_t)(1e9 * (t - (int64_t)t));
  nanosleep(&req, NULL);
}

bool IsDistributedMemoryManager() {
  return NetworkThread::Get()->id() == (NetworkThread::Get()->size() - 1);
}

}
