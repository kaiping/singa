// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-19 15:33
#include <glog/logging.h>

#include "disk/feature_reader.h"
#include "proto/lapis.pb.h"

namespace lapis {

void FeatureReader::Init(const DataSourceProto &ds_proto,
                         const std::vector<std::string> &suffix,
                         int offset) {
  is_.open(ds_proto.path(), std::ofstream::in|std::ofstream::binary);
  CHECK(is_.is_open()) << "Error open the label file "
                       << ds_proto.path() << "\n";
  if (offset > 0) {
    is_.seekg(0, is_.end);
    int length = is_.tellg();
    CHECK(offset < length) << "the offset " << offset
                           << " should be < the file size " << length << "\n";
    is_.seekg(offset, is_.beg);
  }
  // width field is the length for one dimensional feature, see DataSourceProto
  length_ = ds_proto.width();
}

bool FeatureReader::ReadNextRecord(std::string *key, float *val) {
  if (is_.eof())
    return false;
  for (int i = 0; i < length_; i++) {
    is_ >> val[i];
  }
  return true;
}

void FeatureReader::Reset() {
  is_.seekg(0, is_.beg);
}

int FeatureReader::Offset() {
  return is_.tellg();
}

FeatureReader::~FeatureReader() {
  is_.close();
}

}  // namespace lapis
