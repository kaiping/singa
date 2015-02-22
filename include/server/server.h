#ifndef INCLUDE_SERVER_SERVER_H_
#define INCLUDE_SERVER_SERVER_H_
#include "utils/cluster.h"
#include <memory>
using std::shared_ptr;
namespace singa {
class Server{
 public:
  explicit Server(shared_ptr<Cluster> cluster);
  void Run();

 protected:
  shared_ptr<Cluster> cluster_;
};
} /* Server */
#endif //INCLUDE_SERVER_SERVER_H_
