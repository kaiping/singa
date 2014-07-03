// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-01 19:26

#ifndef INCLUDE_DISK_RGB_DIR_READER_H_
#define INCLUDE_DISK_RGB_DIR_READER_H_
#include <string>
#include <vector>
#include "disk/data_reader.h"


namespace lapis {
class RGBDirReader : public DataReaderInterface<string, string> {
 public:
  virtual void init(const DataMetaProto& meta,
                    const std::vector<std::string>& filenames = NULL);
  virtual int next(std::string* k, std::string* v);

 private:
  const vector<string> filenames_;
  const string path_;
  const int height_;
  const int width_;
  const string ending_ = new string(".jpg");
};

}  // namespace lapis

#endif  // INCLUDE_DISK_RGB_DIR_READER_H_
