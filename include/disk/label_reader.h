// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-02 19:45

#ifndef INCLUDE_DISK_LABEL_READER_H_
#define INCLUDE_DISK_LABEL_READER_H_
#include <ifstream>
#include <string>
#include <vector>
#include "disk/record_reader_reader.h"

namespace lapis {
/**
 * Read label from single file, each line consists of filename and labelid
 */
class LabelReader : public RecordReader {
 public:
  // filenames will not be used, and should be NULL
  //
  virtual void Init(const std::string path_prefix,
                    const std::vector<std::string> &path_suffix
                    int offset);

  virtual bool ReadNextRecord(std::string *key, float *val);
  virtual int Offset();
  ~LabelReader();
 private:
  std::ifstream is_;
};

}  // namespace lapis

#endif  // INCLUDE_DISK_LABEL_READER_H_
