// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-11-30 13:30

#ifndef INCLUDE_UTILS_NETWORK_H_
#define INCLUDE_UTILS_NETWORK_H_

#include <google/protobuf/message.h>
#include <memory>
#include <string>
using google::protobuf::Message;

using std::string;
namespace lapis {

/**
 * Network Singleton to transfer data between workers and table servers.
 * It provides the interfaces, e.g., Send, Receive. The implementation can
 * use either MPI, 0MQ or any other libraries.
 */
class Network {
 public:
  /**
   * Underlying network library enumation.
   * Currently only implemented using MPI.
   */
  enum Impl{
    kMPI, //!< MPI library
    kZMQ //!< Zero MQ library
  };
  /**
   * Create a singleton.
   * The created object is a subclass of Network, which implemented the Send
   * Recv functions.
   * @param impl network implementation constant, only kMPI supported currently.
   * @return shared pointer to Network object.
   */
  static std::shared_ptr<Network> Get(Impl impl=kMPI);
  /**
   * Send google protobuf message to remote process.
   * @param dst, ID of the remote process, its semantic depends on the
   * implementation library. E.g., in MPI, it is the rank of the remote process.
   * @param tag, message tag defined by the two communication sides.
   * @param msg, sent data.
   * @return true if send successfully, otherwise false.
   */
  virtual bool Send(int dst, int tag, const string& msg)=0;
  /**
   * Receive google protobuf message.
   * @param tag, the message tag.
   * @parm src, the sending process.
   * @param msg, received data.
   * @return true if the the message is received successfully.
   */
  virtual bool Recv(int *tag, int *src, string* msg)=0;
  /**
   * Send google protobuf message to remote process.
   * @param dst, ID of the remote process, its semantic depends on the
   * implementation library. E.g., in MPI, it is the rank of the remote process.
   * @param tag, message tag defined by the two communication sides.
   * @param msg, google protobuf message.
   * @return true if send successfully, otherwise false.
   */
  virtual bool Send(int dst, int tag, const Message& msg)=0;
  /**
   * Receive google protobuf message.
   * @param tag, message tag, only receive message with this tag.
   * @param msg, pointer to the message to be received.
   * @return sending process ID.
   */
  virtual int Recv(int tag, Message* msg)=0;
 protected:
  Network(){}
  static std::shared_ptr<Network> instance_;
};
class MPINetwork: public Network{
 public:
  virtual bool Send(int dst, int tag, const Message& msg);
  virtual int Recv(int tag, Message* msg);

  virtual bool Send(int dst, int tag, const string& msg);
  virtual	bool Recv(int *tag, int *src, string* msg);
};
} /* lapis  */

#endif
