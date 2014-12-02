// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-11-30 14:16

#include <mpi.h>
#include <glog/logging.h>

#include "utils/network.h"

namespace lapis {

std::shared_ptr<Network> Network::instance_;
std::shared_ptr<Network> Network::Get(Impl impl) {
	if (!instance_) {
		if (impl == kMPI)
			instance_.reset(new MPINetwork());
		else
			LOG(FATAL) << "Network using " << impl << " Not implemented";
	}
	return instance_;
}

bool MPINetwork::Send(int dst, int tag, const Message& msg) {
	std::string buf;
	msg.SerializeToString(&buf);
	MPI_Request req;
	MPI_Isend(const_cast<char*>(buf.data()), buf.size(), MPI::BYTE, dst, tag,
			MPI_COMM_WORLD, &req);
	return true;
}

bool MPINetwork::Recv(int *tag, int *src, Message* msg)=0; {
	MPI_Status status;
	int flag;
	MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status);
	if (!flag) return false;

	*src = status.MPI_SOURCE;
	*tag = status.MPI_TAG;
	std::string buf;
	int count;
	MPI_Get_count(&status, MPI::BYTE, &count);

	buf.resize(count);

	MPI_Recv(const_cast<char*>(buf.data()), count, MPI::BYTE, *src, *tag, &status);
	msg->ParseFromString(buf);
	return true;
}
} /* lapis  */
