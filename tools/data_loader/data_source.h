#ifndef INCLUDE_DATA_SOURCE_H_
#define INCLUDE_DATA_SOURCE_H_
#include <google/protobuf/repeated_field.h>
#include <string>
#include <memory>
#include <map>
#include <vector>
#include <functional>

#include "proto/model.pb.h"

using std::shared_ptr;
using std::vector;
using std::string;
namespace singa {

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
  virtual bool NextRecord(string* key, Record *record)=0;
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

  virtual bool NextRecord(string* key, Record *record);

  //!< class identifier
  static const std::string type;
 protected:
  /**
   * get record at the specific offset
   * return true if succ, otherwise false
   */
  bool GetRecord(const int offset, Record* record);
  /**
   * Read raw image, resize, normalize (substract mean), copy to DAryProto obj
   */
  int ReadImage(const std::string &path, int height, int width,
      const float *mean, DAryProto* datum);
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
  MeanProto data_mean_;
  // record meta info,  a pair of image file name and label
  vector<std::pair<string, int>> lines_;
};
/*****************************************************************************
 * DataSourceFactory
 *****************************************************************************/
/**
 * Register DataSource with identifier ID
 * @param ID identifier of the data source, e.g., "RGBFeature". The id field in
 * DataSourceProto should be the same to this identifier
 * @param DS the child DataSource
 */
#define REGISTER_DATASOURCE(ID, DS) DataSourceFactory::Instance()->\
  RegisterCreateFunction(ID, [](void)-> DataSource* {return new DS();})

/**
 * Factory for creating DataSource instance based on user provided identifier.
 * Users are required to register user-defined DataSource before creating
 * instances of them during runtime through this factory. For example, if you
 * define a new DataSource FooDataSource with identifier "Foo", then you
 * can use it in your net by 1) configure your DataSrouceProto with the id
 * field to be "Foo". 2) register it (e.g., at the start of the program). Then
 * your FooDataSource will be created by calling
 * DataSourceFactory::Instance()->Create("Foo") automatically by the Trainer.
 */

class DataSourceFactory {
 public:
  /**
   * Static method to get instance of this factory, there should be only one
   * instance of this factory.
   */
  static shared_ptr<DataSourceFactory> Instance();

  /**
   * Register user defined DataSource, i.e., add it with its identifier (the
   * id field in DataSourceProto) into a inner map.
   * Later, the factory can then create an instance of the DataSource based on
   * its identifier. This function will be called by the REGISTER_DATASOURCE
   * macro.
   */
  void RegisterCreateFunction(
    const string &id,
    std::function<DataSource*(void)> create_function);
  /**
   * Create an instance the child DataSource of identifier being id.
   * @param id the identifier of the child DataSource.
   */
  DataSource *Create(const string id);

 private:
  //! To avoid creating multiple instances of this factory in the program.
  DataSourceFactory();
  //! Map from DataSource identifier to creating function.
  std::map<const string, std::function<DataSource*(void)>> ds_map_;
  static shared_ptr<DataSourceFactory> instance_;
};

}  // namespace singa
#endif  // INCLUDE_DATA_SOURCE_H_
