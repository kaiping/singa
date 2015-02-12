#include <glog/logging.h>
#include <fcntl.h>
#include <fstream>
#include "utils/cluster.h"
#include "utils/network.h"
#include "proto/cluster.pb.h"
#include "proto/worker.pb.h"

namespace singa {

std::shared_ptr<Cluster> Cluster::instance_;
Cluster::Cluster(const ClusterProto &cluster, string hostfile, int procsID) {
  procsID_=procsID;
	cluster_ = cluster;
  SetupFolders(cluster);
  char hostname[256];
  gethostname(hostname, sizeof(hostname));
  hostname_.insert(0, hostname, sizeof(hostname));

  std::ifstream ifs(hostfile, std::ifstream::in);
  while(std::getline(ifs, line)){
    addr_.push_back(line);
  }
  CHECK_EQ(addr_.size(), cluster_.nservers()+cluster_.nworkers());
}

void Cluster::SetupFolders(const ClusterProto &cluster){
  // create visulization folder
  mkdir(visualization_folder().c_str(),  S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
}

void Cluster::SetupGroups(const ClusterProto &cluster){
  char tmp[256];
  int len;
}

shared_ptr<Cluster> Cluster::Get(const ClusterProto& cluster, string hostfile,
    int procsID){
  if(!instance_) {
    instance_.reset(new Cluster(cluster, hostfile, procsID));
  }
  return instance_;
}

shared_ptr<Cluster> Cluster::Get() {
  if(!instance_) {
    LOG(ERROR)<<"The first call to Get should "
              <<"provide the sys/model conf path";
  }
  return instance_;
}
}  // namespace singa
