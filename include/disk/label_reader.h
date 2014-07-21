// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-02 19:45

#ifndef INCLUDE_DISK_LABEL_READER_H_
#define INCLUDE_DISK_LABEL_READER_H_
#include <fstream> // #NOLINT
#include <string>
#include <vector>
#include "disk/record_reader.h"

namespace lapis {
/**
 * Read label from single file, each line consists of filename and labelid
 */
class LabelReader : public RecordReader {
 public:
  // filenames will not be used, and should be NULL
  virtual void Init(const DataSourceProto &ds_proto,
                    const std::vector<std::string> &path_suffix,
                    int offset = 0);

  /**
   * Read the label for the next record.
   * The label is of type int in the file, hence we have to convert it into
   * float, and set val.
   * @param key the identifier of the record, e.g., suffix of the path of the
   * image file.
   * @param val the label
   */
  virtual bool ReadNextRecord(std::string *key, float *val);
  virtual void Reset();
  virtual int Offset();
  virtual std::string id() {return id_;}
  ~LabelReader();
 private:
  static const std::string id_;
  std::ifstream is_;
};

}  // namespace lapis

#endif  // INCLUDE_DISK_LABEL_READER_H_
