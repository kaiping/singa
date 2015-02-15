#ifndef INCLUDE_UTILS_ROUTER_H_
#define INCLUDE_UTILS_ROUTER_H_
#include <czmq.h>
#include <string>
#include <vector>
using std::pair;
using std::vector;
using std::string;
namespace singa {

#define HEADER "singa"

class Router{
 public:
  explicit Router(int port);
  ~Router();
  /**
   * block operation, ping-pong with the server to confirm a connection
   * is established.
   */
  bool Connect(std::string addr);

  bool Bind(std::string addr, int expected_connections);

  /**
   * push identifier, header and signature of sender
   * send, then delete
   */
  void Send(zmsg_t* msg, int serverid);

  /**
   * reply to last sender, push identifier, header and signature
   */
  void Reply(zmsg_t* msg);

  /**
   * pop identifier, header and signature
   * upper app delete the msg.
   */
  zmsg_t* Recv();

 protected:
  int port_;
  zsock_t *router_;
  // endpoint, sequence
  std::vector<std::pair<string, int64_t>> nodes_;
  char* last_recv_node_;
};
} /* singa */
#endif // INCLUDE_UTILS_ROUTER_H_
