// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-11-30 14:16

#include <mpi.h>
#include "utils/network.h"

namespace lapis {

std::shared_ptr<Network> Network::instance_;
std::shared_ptr<Network> Network::Get(Impl impl){
  if(!instance_){
    if(impl==kMPI)
      instance_.reset(new MPINetwork());
    else
      LOG(FATAL)<<"Network using "<<impl<<" Not implemented";
  }
  return instance_;
}


bool MPINetwork::Send(int dst, int tag, const Message& msg) {
  std::string buf=msg.SerializeToString();
  MPI_Request req;
  MPI_ISend(buf.data(), buf.size(), MPI::BYTE, dst, tag, MPI_COMM_WORLD, &req);
  return true;
}

int MPINetwork::Recv(int tag, Message* msg){
  MPI_Status status;
  MPI_Message mpi_msg;
  MPI_MProbe(MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, mpi_msg, &status);
  int count;
  MPI_Get_count(&status, MPI::BYTE, count);
  std::string buf;
  buf.resize(count);
  MPI_Mrecv(buf.data(), count, MPI::BYTE, mpi_msg, &status);
  return status.MPI_SOURCE;
}
} /* lapis  */
