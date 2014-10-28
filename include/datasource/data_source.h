// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-16 22:00

#ifndef INCLUDE_DISK_DATA_SOURCE_H_
#define INCLUDE_DISK_DATA_SOURCE_H_
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
namespace lapis {
typedef vector<string> StringVec;
/**
 * Base class of data source which provides training records for applications.
 * Every worker will call it to parse raw data and insert records into leveldb
 */
class DataSource {
 public:
  virtual ~DataSource(){}
  virtual void Init(const DataSourceProto &ds_proto);
  virtual void Init(const DataSourceProto &proto, const ShardProto& shard)=0;
  virtual void ToProto(DataSourceProto *ds_proto);
  virtual int NextRecord(string* key, Record *record)=0;
  void Next() {offset_++;}
  //virtual bool GetRecord(const int key, Record* record)=0;
  //virtual void Reset(const ShardProto& sp)=0;
  /**
   * Return name of this data source
   */
  const string &name() {
    return name_;
  }
  /**
   * Return number of instances of this data source.
   */
  const int size() {
    return size_;
  }
  /**
   * Return the offset (or id) of current record to the first record.
   * This offset may be used by the Trainer to check whether all data sources
   * are in sync.
   */
  int offset() {
    return offset_;
  }

  bool eof() {
    return offset_>=size_;
  }

 protected:
  //! total number of instances/images
  int size_;
  //! offset from current record to the first record
  int offset_;
  //! identifier of the data source
  string name_;
  //! data source type
  // DataSourceProto_DataType type_;
};
/**
 * DataSource whose local source is rgb images under a directory.
 */
class ImageNetSource : public DataSource {
 public:
  virtual void Init(const DataSourceProto &proto, const ShardProto& shard);
  virtual void Init(const DataSourceProto &proto);
  virtual int NextRecord(string* key, Record *record);

  void Reset(const ShardProto& sp);
  bool GetRecord(const int key, Record* record);
  int ReadImage(const std::string &path, int height, int width,
      const float *mean, DAryProto* datum);
  void LoadLabel(string path);
  void LoadMeanFile(string path);
  int CopyFileFromHDFS(string hdfs_path, string local_path) ;
  int CopyFilesFromHDFS(string hdfs_folder, string local_folder,
    std::vector<string> files) ;
  //! the identifier, i.e., "RGBSource"
  static const std::string type;

 private:
  bool hdfs_; // if true, all following paths are hdfs path
  /**
   * image folder + lines[i].first is the full path of one image
   */
  std::string image_folder_;
  std::string label_path_;
  std::string mean_file_;
  /**
   * expected height of the image, assume all images are of the same shape; if
   * the real shape is not the same as the expected, then resize it. The
   */
  int height_;
  //! width of the image, assume all images are of the same shape
  int width_;
  // should be fixed to 3, because this is rgb feature.
  int channels_;
  int record_size_;
  bool do_shuffle_;
  MeanProto *data_mean_;
  //! pairs of image file name and label
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

}  // namespace lapis

#endif  // INCLUDE_DISK_DATA_SOURCE_H_
