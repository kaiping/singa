// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-02 19:50

#include <glog/logging.h>

#include "disk/label_reader.h"
namespace lapis {

void LabelReader::Init(const DataSourceProto &ds_proto,
                       const vector<std::string> &suffix,
                       int offset = 0) {
  is_.open(ds_proto.prefix);
  CHECK(is_.is_open())<<"Error open the label file "<<prefix<<"\n";
  if (offset>0) {
    is_.seekg(0, is.end);
    int length=is_.tellg();
    CHECK_LE(offset, length) << "the offset "<<offset
                             <<" should be < the file size "<<length<<"\n";
    is_.seekg(offset, is_.beg);
  }
}

bool LabelReader::ReadNextRecord(std::string *key, float *val) {
  if (is_.eof())
    return false;
  int v;
  fin_ >> *k >> v;
  *val= (float) v;
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
