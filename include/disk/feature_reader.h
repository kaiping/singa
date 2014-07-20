// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-19 15:22
#ifndef INCLUDE_DISK_FEATURE_READER_H_
#define INCLUDE_DISK_FEATURE_READER_H_
#include <vector>
#include <string>
#include <fstream>

#include "disk/record_reader.h"

namespace lapis {
/**
 * Read normal feature data.
 * Each record/feature is a one line vector in the single input file.
 */
class FeatureReader : public RecordReader {
 public:
  /**
   * Initialization.
   * @param ds_proto since there is only one input file, we only need the
   * file path, i.e., the path_prefix from DataSourceProto.
   * @param path_suffix not used
   * @param offset the offset to the beginning of the input file (not in terms
   * of lines, it is computed by the ifstream::tellg()).
   */
  virtual void Init(const DataSourceProto &ds_proto,
                    const std::vector<std::string> &path_suffix,
                    int offset = 0);

  /**
   * Read next feature.
   * @param key null or set to the line id
   * @param val put the feature to val
   */
  virtual bool ReadNextRecord(std::string *key, float *val);
  /**
   * Reset to read from the beginning of the file
   */
  virtual void Reset();
  /**
   * Return the offset in terms of position in the ifstream.
   */
  virtual int Offset();
  ~FeatureReader();

 private:
  /**
   * length of the feature, should be configured at the width field in
   * DataSourceProto
   */
  int length_;
  std::ifstream is_;
};

}  // namespace lapis

#endif  // INCLUDE_DISK_FEATURE_READER_H_
