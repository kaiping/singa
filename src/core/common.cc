#include "core/common.h"
#include "core/file.h"
#include "static-initializers.h"
#include "core/rpc.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <signal.h>
#include <execinfo.h>
#include <fcntl.h>

#include <math.h>

#include <asm/msr.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <lzo/lzo1x.h>
#include <mpi.h>

DEFINE_bool(localtest, false, "");
DEFINE_bool(run_tests, false, "");

DEFINE_string(hostfile, "conf/mpi-beakers", "");
DEFINE_int32(workers, 2, "");

namespace lapis {

void Sleep(double t) {
  timespec req;
  req.tv_sec = (int)t;
  req.tv_nsec = (int64_t)(1e9 * (t - (int64_t)t));
  nanosleep(&req, NULL);
}

//  this is called once for every MPI process
DistributedMemoryManager* InitServers(int argc, char** argv) {
  FLAGS_logtostderr = true;
  FLAGS_logbuflevel = -1;

  google::SetUsageMessage("%s: invoke from mpirun, using --runner to select control function.");
  google::ParseCommandLineFlags(&argc, &argv, false);
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();

  //  no initializer registered
  RunInitializers();

  // Assumed that we launch using MPI command
  /*
  if (!getenv("OMPI_UNIVERSE_SIZE")) {
    string cmd = StringPrintf("mpirun "
                              " -hostfile %s"
                              " -bycore"
                              " -nooversubscribe"
                              " -n %d"
                              " %s"
                              " --log_prefix=false ",
                              FLAGS_hostfile.c_str(),
                              FLAGS_workers,
                              JoinString(&argv[0], &argv[argc]).c_str()
                              );

    LOG(INFO) << "Invoking MPI..." << cmd;
    system(cmd.c_str());
    exit(0);
  }
  */

  //  start network thread for this process
  NetworkThread::Init();
  NetworkThread* net = NetworkThread::Get();

  //  return DistributedMemoryManager for rank size()-1
  //  else, initialize MemoryServer and return null;
  if (net->id()==(net->size()-1))
	  return DistributedMemoryManager::Get();
  else{
	  (new MemoryServer())->Init();
	  return NULL;
  }
}

}
