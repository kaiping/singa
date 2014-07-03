// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-06-28 19:57

#ifndef INCLUDE_DISK_DATA_READER_H_
#define INCLUDE_DISK_DATA_READER_H_

#include <string>
#include <vector>
#include "proto/lapis.pb.h"

namespace lapis {
// this is the interface for reading training data.
// users have to implement this interface and register
// their own data reader class with a name in the main function
// we require labelled data should have a key e.g., filename of the image,
// for both label source and feature source
class DataReaderInterface {
 public:
  // initialization
  // meta provides the path, which is a file path if all data is in a single
  // file; it is also possible that path is a directory, e.g., under which are
  // image files.
  // if the path is a directory, then for each filename path+filename can
  // locate an image. The filenames can be extracted from the label data
  // source, e.g., the label source of imagenet is like: <image filename,
  // labelid>, and the feature source is a directory (i.e., the path)
  // contains all images.
  virtual void init(const DataMetaProto* meta,
                    const std::vector<std::string>& filenames = NULL) = 0;
  // return true if not at the end of reading, otherwise return false
  // the record is parsed and serialized into k and v
  virtual bool next(std::string* k, std::string* v) = 0;
};
}  // namespace lapis

#endif  // INCLUDE_DISK_DATA_READER_H_
