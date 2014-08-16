// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-02 19:45

#ifndef INCLUDE_DISK_LABEL_SOURCE_H_
#define INCLUDE_DISK_LABEL_SOURCE_H_
#include <fstream> // #NOLINT
#include <string>
#include <vector>
#include <memory>

#include "datasource/data_source.h"
#include "net/lapis.h"


namespace lapis {
/**
 * Read label from single file, each line consists of filename and labelid
 * Test with imagenet data.
 */
class LabelSource : public DataSource {
 public:
  /**
   * Init this data source, get the label file path from ds_proto
   * @param ds_proto the user configured data source meta info
   */
  const std::shared_ptr<StringVec> Init(const DataSourceProto &ds_proto,
      std::shared_ptr<StringVec>& filenames);
  virtual void ToProto(DataSourceProto *proto);
  /**
   * Fill the blob with labels. If rich end of the label list, then repeat from
   * the beginning.
   * @param blob the blob to be filled. Since we have read all labels into
   * memory, this function just copy the labels to the blob
   */
  virtual void GetData(Blob *blob);
  virtual void NextRecord(FloatVector* record);
  /**
   * Single label file is small, this function do load them into memory; and
   * put them to distributed disk depending on the availability of the
   * distributed disk.
   */
  virtual const std::shared_ptr<StringVec> LoadData(
    const std::shared_ptr<StringVec>  &keys);

  /**
   * channel, height, width are all assumed to be 1.
   */
  virtual int channels() {
    return 1;
  }
  virtual int height() {
    return 1;
  }
  virtual int width() {
    return 1;
  }
  /*
  virtual bool has_channels() {return true;}
  virtual bool has_height() {return true;}
  virtual bool has_width() {return true;}
  */
  static const std::string type;

 private:
  //! path for the label file
  std::string label_path_;
  /**
   * the suffix paths for all images, which can be used to identify the images
   * make it a shared pointer for passing it to other data sources to locate
   * the data file, e.g., when loading rgb images.
   */
  std::shared_ptr<StringVec> image_names_;
  //! labels, converted to float for Blob
  std::vector<float> labels_;
};
}  // namespace lapis

#endif  // INCLUDE_DISK_LABEL_SOURCE_H_
