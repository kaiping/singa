// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-02 19:50

namespace lapis {
void LabelDirReader::init(const DataMetaProto& meta,
                          const vector<string>& filenames = NULL) {
  fin_ = new std::ifstream(meta.path);
}

int LabelDirReader::next(string* k, string* v) {
  if (fin_.eof())
    return 0;

  fin_>>*k>>*v;
  return 1;
}

LabelDirReader::~LabelDirReader() {
  fin_.close();
  delete fin_;
}
}  // namespace lapis
