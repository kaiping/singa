// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-02 19:50

#include "disk/label_reader.h"
namespace lapis {
void LabelReader::init(const std::string &prefix,
                       const vector<std::string> &suffix) {
  fin_ = new std::ifstream(prefix);
  CHECK(fin_->good())<<"Error open the label file "<<prefix<<"\n";
}

bool LabelReader::ReadNextRecord(std::string *key, float *val) {
  if (fin_->eof())
    return 0;

  fin_ >> *k >> *v;
  return 1;
}

LabelReader::~LabelReader() {
  fin_.close();
  delete fin_;
}
}  // namespace lapis
