#ifndef INCLUDE_UTILS_ROUTER_H_
#define INCLUDE_UTILS_ROUTER_H_
#include <czmq.h>
#include <string>
#include <vector>
using std::pair;
using std::vector;
using std::string;
namespace singa {

#define HEADER "singa\0"

/**
 * for communicating between Server and ParameterManager.
 */
class Router{
 public:
  explicit Router(int port);
  ~Router();
  /**
   * block operation, ping-pong with the server to confirm a connection
   * is established.
   */
  bool Connect(std::string addr);

  bool Bind(std::string addr, size_t expected_connections);

  /**
   * push identifier, header and signature of sender
   * send, then delete
   */
  void Send(zmsg_t* msg, int serverid);

  /**
   * reply to last sender, push identifier, header and signature
   */
  void Reply(zmsg_t* msg);
  void Reply(zmsg_t* msg, string endpoint);
  void Reply(zmsg_t* msg, zframe_t* identity);

  /**
   * pop identifier, header and signature
   * upper app delete the msg.
   */
  zmsg_t* Recv();

  zsock_t* router() const {
    return router_;
  }

 protected:
  int port_;
  zsock_t *router_;
  zframe_t* last_recv_node_;
  // endpoint, sequence
  std::vector<std::pair<string, int64_t>> nodes_;
};
} /* singa */
#endif // INCLUDE_UTILS_ROUTER_H_
