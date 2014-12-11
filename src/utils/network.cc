// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-11-30 14:16

#include <mpi.h>
#include <glog/logging.h>

#include "utils/network.h"
#include <cstdio>

namespace lapis {

bool MPINetwork::Send(int dst, int tag, const string& msg) {
	MPI_Request req;
	if (tag==32 && msg.size()==15){
		for (int i=0; i<msg.size(); i++)
			std::printf("%02x ",msg[i]);
		std::printf("\n");
	}
	/*MPI_Isend(const_cast<char*>(msg.data()), msg.size(), MPI::BYTE, dst, tag,
			MPI_COMM_WORLD, &req);*/
	MPI_Send(const_cast<char*>(msg.data()), msg.size(), MPI::BYTE, dst, tag,
				MPI_COMM_WORLD);
	return true;
}

bool MPINetwork::Recv(int *tag, int *src, string* msg){
	MPI_Status status;
	int flag;
	MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status);
	if (!flag) return false;

	*src = status.MPI_SOURCE;
	*tag = status.MPI_TAG;

	int count;
	MPI_Get_count(&status, MPI::BYTE, &count);

	msg->resize(count);
	MPI_Recv(const_cast<char*>(msg->data()), count, MPI::BYTE, *src, *tag, MPI_COMM_WORLD, &status);
	return true;
}

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

} /* lapis  */
