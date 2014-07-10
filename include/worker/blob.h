// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-06 14:13

#ifndef INCLUDE_WORKER_BLOB_H_
#define INCLUDE_WORKER_BLOB_H_
#include <vector>
#include <string>

#include "proto/lapis.proto.h"

namespace lapis {
class Blob {
 public:
  explicit Blob(const BlobProto& blob_proto);
  void Init();  // allocate memory
 private:
  std::vector<float> content_;
  std::string name_;
};

}  // namespace lapis
#endif  // INCLUDE_WORKER_BLOB_H_

