// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-10 22:29

#ifndef INCLUDE_WORKER_EDGE_H_
#define INCLUDE_WORKER_EDGE_H_
namespace lapis {
class Edge {
 public:
  explicit Edge(const EdgeProto& edge_proto);
  void GetInput(const DataMeta& data_source);
 private:
  Param* param_ = nullptr;
  Blob* data_ = nullptr;
};

}  // namespace lapis
#endif  // INCLUDE_WORKER_EDGE_H_
