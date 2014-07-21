// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-20 16:11

#ifndef INCLUDE_DISK_IDX_READER_H_
#define INCLUDE_DISK_IDX_READER_H_
#include <vector>
#include <string>
#include <fstream> // #NOLINT
#include "proto/lapis.pb.h"
#include "disk/record_reader.h"
namespace lapis {
/**
 * Reader for IDX format file.
 * The IDX format is documented at the
 * <a href="http://yann.lecun.com/exdb/mnist/"> mnist page </a>.
 * This reader is mainly to read the mnist data.
 */

class IDXReader : public RecordReader {
 public:
  /**
   * Init the reader.
   * Read file path from the DataSourceProto. Open the file, and parse the
   * magic number.
   * @param ds_proto DataSourceProto configed by user.
   * @param path_suffix empty vector.
   * @param offset The offset of the next record to the first record. If >0
   * then the reader is restored from some checkpoint.
   */
  virtual void Init(const DataSourceProto &ds_proto,
                    const std::vector<std::string> &path_suffix,
                    int offset = 0);

  /**
   * Read the feature for next record.
   * @param key set to the id of the record
   * @param val the content of the feature of the record
   */
  virtual bool ReadNextRecord(std::string *key, float *val);
  /**
   * Reset to read from the first record
   */
  virtual void Reset();
  /**
   * Return the offset from the current record to the first record.
   */
  virtual int Offset();
  virtual std::string id() { return id_;}
  ~IDXReader();

  enum IDXDataType {
    kIDX_ubyte = 0x08,
    kIDX_byte = 0x09,
    kIDX_short = 0x0B,
    kIDX_int = 0x0C,
    kIDX_float = 0x0D,
    kIDX_double = 0x0E
  };

 private:
  /**
   * Identifier of this reader, i.e., "IDXReader"
   */
  static const std::string id_;

  std::ifstream is_;
  //! record length
  int length_;
  //! record size in terms of bytes
  int num_bytes_;
  //! data type
  int type_;
  //! buffer for the read function
  char *buf_;
};

}  // namespace lapis

#endif  // INCLUDE_DISK_IDX_READER_H_

