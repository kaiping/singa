// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-02 19:50

#include <glog/logging.h>
#include "disk/label_reader.h"
namespace lapis {
const std::string IDXReader::id_="IDXReader";
void LabelReader::Init(const DataSourceProto &ds_proto,
                       const std::vector<std::string> &suffix,
                       int offset) {
  is_.open(ds_proto.path());
  CHECK(is_.is_open()) << "Error open the label file "
                       << ds_proto.path() << "\n";
  if (offset > 0) {
    is_.seekg(0, is_.end);
    int size = is_.tellg();
    CHECK_LE(offset, size) << "the offset " << offset
                           << " should be < the file size " << size << "\n";
    is_.seekg(offset, is_.beg);
  }
}

bool LabelReader::ReadNextRecord(std::string *key, float *val) {
  if (is_.eof())
    return false;
  int v;
  is_ >> *key >> v;
  *val = static_cast<float> (v);
  return true;
}

void LabelReader::Reset() {
  is_.seekg(0, is_.beg);
}

int LabelReader::Offset() {
  return is_.tellg();
}

LabelReader::~LabelReader() {
  is_.close();
}
}  // namespace lapis
