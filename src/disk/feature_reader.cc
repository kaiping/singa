// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-19 15:33

#include "disk/feature_reader.h"

void FeatureReader::Init(const DataSourceProto &ds_proto,
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
  //! in case that the user write the length at wrong field
  length_=ds_proto.width()*ds_proto.height()*ds_proto.channels();
}

bool FeatureReader::ReadNextRecord(std::string *key, float *val) {
  if (is_.eof())
    return false;
  for(int i=0;i<length_;i++) {
    is_>>val[i];
  }
  return true;
}


void FeatureReader::Reset() {
  is_.seekg(0, is_.beg);
}

int LabelReader::Offset() {
  return is_.tellg();
}

FeatureReader::~FeatureReader() {
  is_.close();
}
