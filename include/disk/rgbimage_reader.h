// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-01 19:26

#ifndef INCLUDE_DISK_RGBIMAGE_READER_H_
#define INCLUDE_DISK_RGBIMAGE_READER_H_

#include <string>
#include <vector>
#include "disk/record_reader.h"


namespace lapis {
class RGBImageReader : public RecordReader {
 public:
  virtual void Init(const DataSourceProto &ds_proto,
                    const std::vector<std::string> &path_suffix,
                    int offset = 0);
  /**
   * Read next image.
   * @param key the path suffix that can identify this image
   * @param val the content of the rgb feature for the next image
   * @return true if read success; false otherwise
   */
  virtual bool ReadNextRecord(std::string *key, float *val);
  /**
   * Reset to read from the beginning.
   * Sometimes we need to go through the dataset multiple times.
   */
  virtual void Reset();
  virtual int Offset();

 private:
  /**
   * common prefix of all images, join(path_prefix_,path_suffix) is full image
   * path.
   */
  string path_prefix_;
  std::vector<std::string> path_suffix_;
  /**
   * expected height of the image, assume all images are of the same shape; if
   * the real shape is not the same as the expected, then resize it. The
   * channels is fixed to be 3, because this is rgb image.
   */
  int height_;
  //! width of the image, assume all images are of the same shape
  int width_;
  //! offset of the next image to read in the path_suffix_ list
  std::vector<std::string>::iterator it;
};

}  // namespace lapis

#endif  // INCLUDE_DISK_RGBIMAGE_READER_H_
