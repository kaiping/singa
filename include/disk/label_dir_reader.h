// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-02 19:45

#ifndef INCLUDE_DISK_LABEL_DIR_READER_H_
#define INCLUDE_DISK_LABEL_DIR_READER_H_
#include <ifstream>
#include <string>
#include <vector>
#include "disk/data_reader.h"

namespace lapis {
// read label from single file, each line is image filename+labelid
class LabelDirReader : public DataReaderInterface<string, string> {
 public:
  // filenames will not be used, and should be NULL
  virtual void init(const DataMetaProto &meta,
                    const std::vector<std::string> &filenames = NULL);
  virtual int next(string *k, string *v);
 private:
  ~LabelDirReader();
  const std::string path_;
  std::ifstream *fin_;
};

}  // namespace lapis

#endif  // INCLUDE_DISK_LABEL_DIR_READER_H_
