// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-16 22:00

#ifndef INCLUDE_MODEL_DATA_SOURCE_H_
#define INCLUDE_MODEL_DATA_SOURCE_H_
#include <string>

#include "model/blob.h"
#include "proto/lapis.pb.h"


namespace lapis {
class DataSource {
 public:
  explicit DataSource(const DataSourceProto &ds_proto);
  void ToProto(DataSourceProto *ds_proto);
  /**
   * Put one batch is data into blob
   * @param blob where the next batch of data will be put
   */
  void GetData(Blob *blob);
  const int Batchsize() {
    return batchsize_;
  }
  const int Size() {
    return size_;
  }
  const int Channels() {
    return channels_;
  }
  const int Height() {
    return height_;
  }
  const int Width() {
    return width_;
  }
  const std::string &Name() {
    return name_;
  }

 private:
  //! current batch id
  int batchid_;
  //! num of instances per batch
  int batchsize_;
  //! total number of instances/images
  int size_;
  //! data source type
  DataSourceProto_DataType type_;
  //! the path of the data source file
  std::string path_;
  //! identifier of the data source
  std::string name_;
  //! parser name for this data source
  std::string parser_;
  //! properties for rgs image feature
  int channels_, height_, width_;
};

}  // namespace lapis

#endif  // INCLUDE_MODEL_DATA_SOURCE_H_
