// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-01 19:26

#ifndef INCLUDE_DISK_RGB_DIR_SOURCE_H_
#define INCLUDE_DISK_RGB_DIR_SOURCE_H_

#include <string>
#include <vector>
#include <memory>

#include "datasource/data_source.h"
#include "net/lapis.h"
#include "proto/model.pb.h"


namespace lapis {
/**
 * DataSource whose local source is rgb images under a directory.
 */
class RGBDirSource : public DataSource {
 public:
  const std::shared_ptr<StringVec> Init(const DataSourceProto &ds_proto,
      std::shared_ptr<StringVec>& filenames);
  virtual void GetData(Blob *blob);
  virtual void NextRecord(FloatVector* record);

  /**
   * Load rgb images.
   * Do nothing except setting the suffix paths for images, if load to single
   * machine memory; Otherwise, read images and put them into distributed disk.
   * @param keys pointer to a vector of suffix paths for image files, can be
   * nullptr
   * @return pointer to a vector of suffix paths for image files found by this
   * function
   */
  virtual const std::shared_ptr<StringVec> LoadData(
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

  //! the identifier, i.e., "RGBSource"
  static const std::string type;

 private:
  /**
   * common prefix of all images, join(directory_,path_suffix) is full image
   * path.
   */
  std::string directory_;
  /**
   * expected height of the image, assume all images are of the same shape; if
   * the real shape is not the same as the expected, then resize it. The
   */
  int height_;
  //! width of the image, assume all images are of the same shape
  int width_;
  // should be fixed to 3, because this is rgb feature.
  int channels_;
  //! image size in terms of floats
  int image_size_;
  std::string mean_file_;
  MeanProto data_mean_;
  //! names of images under the directory_
  std::shared_ptr<StringVec> image_names_;
};

}  // namespace lapis

#endif  // INCLUDE_DISK_RGB_DIR_SOURCE_H_
