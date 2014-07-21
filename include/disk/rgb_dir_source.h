// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-01 19:26

#ifndef INCLUDE_DISK_RGB_DIR_SOURCE_H_
#define INCLUDE_DISK_RGB_DIR_SOURCE_H_

#include <string>
#include <vector>
#include <memory>

#include "disk/data_source.h"
#include "model/blob.h"

namespace lapis {
/**
 * DataSource whose local source is rgb images under a directory.
 */
class RGBDirSource : public DataSource {
 public:
  virtual void Init(const DataSourceProto &ds_proto);
  virtual void GetData(Blob *blob);
  /**
   * Load rgb images.
   * Do nothing except setting the suffix paths for images, if load to single
   * machine memory; Otherwise, read images and put them into distributed disk.
   * @param keys pointer to a vector of suffix paths for image files, can be
   * nullptr
   * @return pointer to a vector of suffix paths for image files found by this
   * function
   */
  virtual const std::shared_ptr<StringVec> &LoadData(
    const std::shared_ptr<StringVec>  &keys);
  virtual int channels() {
    return 3;
  }
  virtual int height() {
    return height_;
  }
  virtual int width() {
    return width_;
  }
  virtual bool has_channels() {
    return true;
  }
  virtual bool has_height() {
    return true;
  }
  virtual bool has_width() {
    return true;
  }
  virtual const std::string &id() {
    return id_;
  }

  //! the identifier, i.e., "RGBSource"
  static const std::string id_;

 private:
  /**
   * common prefix of all images, join(directory_,path_suffix) is full image
   * path.
   */
  std::string directory_;
  /**
   * expected height of the image, assume all images are of the same shape; if
   * the real shape is not the same as the expected, then resize it. The
   * channels is fixed to be 3, because this is rgb image.
   */
  int height_;
  //! width of the image, assume all images are of the same shape
  int width_;
  //! size of one record in terms of floats
  int record_length_;
  //! names of images under the directory_
  std::shared_ptr<StringVec> image_names_;
};

}  // namespace lapis

#endif  // INCLUDE_DISK_RGB_DIR_SOURCE_H_
