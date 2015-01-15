#include <google/protobuf/repeated_field.h>
#include <string>
#include <memory>
#include <map>
#include <vector>
#include <functional>
#include <fstream>

#include "proto/model.pb.h"

using std::shared_ptr;
using std::vector;
using std::string;

/**
 * Base class for data sources.
 * It defines the APIs for generating key-value tuples by paring raw data
 * (e.g., images). It is used in creating data Shards.
 */
class DataSource {
 public:
  DataSource() : size_(0), offset_(0), name_("unknown"){}
  /**
   * Fetch/parse next record.
   * @key pointer to key string, exist content will be overwrite
   * @pointer to Record, exist content will be overwrite
   * @return true if read succ, false otherwise
   */
  virtual bool NextRecord(string* key, singa::Record *record)=0;
  /**
   * @return name of this data source
   */
  const string &name() {
    return name_;
  }
  /**
   * @return number of instances of this data source
   */
  const int size() {
    return size_;
  }
  /**
   * @return the offset (or id) of current record to the first record.
   */
  int offset() {
    return offset_;
  }
  /**
   * @return true if reaches the end of parsing
   */
  bool eof() {
    return offset_>=size_;
  }

 protected:
  //!< total number of instances/images
  int size_;
  //!< offset from current record to the first record
  int offset_;
  //!< identifier of the data source
  string name_;
};
class MnistSource :public DataSource{
 public:
  void Init(string imagefile, string labelfile);
  /**
   * Fetch/parse next record.
   * @key pointer to key string, exist content will be overwrite
   * @pointer to Record, exist content will be overwrite
   * @return true if read succ, false otherwise
   */
  virtual bool NextRecord(string* key, singa::Record *record);
  /**
   * @return name of this data source
   */
  const string &name() {
    return name_;
  }
 protected:
  char* image_;
  int height_, width_;
  std::ifstream imagestream_, labelstream_;
};

/**
 * ImageNet dataset specific source.
 */
class ImageNetSource : public DataSource {
 public:
  /**
   * @folder local shard folder for train/validation/test.It's subdirs/files
   * include img/ for original images; rid.txt for record meta info (image path
   * and label pair); shard.dat, storing parsed records;
   * @meanfile, mean google protobuf file for the imagenet images
   * @width, resize images to this width
   * @height, resize images to this height
   */
  void Init(const string& folder, const string& meanfile,
      const int width, const int height);

  virtual bool NextRecord(string* key, singa::Record *record);

 protected:
  /**
   * get record at the specific offset
   * return true if succ, otherwise false
   */
  bool GetRecord(const int offset, singa::Record* record);
  /**
   * Read raw image, resize, normalize (substract mean), copy to DAryProto obj
   */
  int ReadImage(const std::string &path, int height, int width,
      const float *mean, singa::DAryProto* datum);
  /**
   * Load meta info file which has image path and label
   */
  void LoadLabel(string path);
  /**
   * Load meanfile
   */
  void LoadMeanFile(string path);

 private:
  /**
   * image folder (shard_folder/img) + lines[i].first is the full path of one image
   *
   */
  std::string image_folder_;
  /**
   * label_path_ is shard_folder/rid.txt
   */
  std::string label_path_;
  std::string mean_file_;
  /**
   * expected height of the image, assume all images are of the same shape; if
   * the real shape is not the same as the expected, then resize it.
   */
  int height_;
  //! width of the image, assume all images are of the same shape
  int width_;
  /**
   * resized image size
   */
  int record_size_;
  singa::MeanProto data_mean_;
  // record meta info,  a pair of image file name and label
  vector<std::pair<string, int>> lines_;
};

