// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-16 22:00

#ifndef INCLUDE_DISK_DATA_SOURCE_H_
#define INCLUDE_DISK_DATA_SOURCE_H_
#include <string>
#include <memory>
#include <map>
#include <vector>
#include "model/blob.h"
#include "proto/lapis.pb.h"


namespace lapis {
typedef std::vector<std::string> StringVec;
/**
 * Base class to data source.
 * It has 2 tasks. One is to fill the blob for DataLayer either from local disk
 * or distributed disk; Another one is for the Coordinator to load data from
 * local disk to distributed disk.
 */
class DataSource {
 public:
  virtual void Init(const DataSourceProto &ds_proto);
  virtual void ToProto(DataSourceProto *ds_proto);
  /**
   * Put one batch data into blob, the blob will specify the num of instances
   * to read (the blob is setup by layer Setup(), which has the batchsize as
   * one parameter).The datasource itself know the size of each record
   * (from DataSourceProto). This function will either read data from disk file
   * or distributed disk.
   * @param blob where the next batch of data will be put
   */
  virtual void GetData(Blob *blob) = 0;
  /**
   * TODO(wnagwei) Load data and return the keys of all records.
   * if the distributed disk is not available, it will be loaded into single
   * machine (i.e., memory), e.g., labels. It is also possible the this
   * function does nothing except do some settings for reading records from
   * disk by GetData(). If the distributed disk is available, it will load data
   * onto distributed disk.
   * @param keys it specifies the order of records to read.
   * @return the keys of records which specifies the order records read.
   */
  virtual const std::shared_ptr<StringVec> &LoadData(
    const std::shared_ptr<StringVec>  &keys) = 0;

  /**
   * Return the identifier of the DataSource
   */
  virtual const std::string &id() = 0;

  /**
   * Return the number of channels, e.g., 3 for rgb data
   */
  virtual int channels() = 0;
  /**
   * Return the height of the record, e.g., the height of an image
   */
  virtual int height() = 0;
  /**
   * Return the width of the record, e.g., the width of an image
   */
  virtual int width() = 0;
  /*
  virtual bool has_channels()=0;
  virtual bool has_height()=0;
  virtual bool has_width()=0;
  */

  /**
   * Return name of this data source
   */
  const std::string &name() {
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

 protected:
  //! total number of instances/images
  int size_;
  //! offset from current record to the first record
  int offset_;
  //! identifier of the data source
  std::string name_;
  //! data source type
  // DataSourceProto_DataType type_;
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
  static DataSourceFactory *Instance();

  /**
   * Register user defined DataSource, i.e., add it with its identifier (the
   * id field in DataSourceProto) into a inner map.
   * Later, the factory can then create an instance of the DataSource based on
   * its identifier. This function will be called by the REGISTER_DATASOURCE
   * macro.
   */
  void RegisterCreateFunction(
    const std::string &id,
    std::function<DataSource*(void)> create_function);
  /**
   * Create an instance the child DataSource of identifier being id.
   * @param id the identifier of the child DataSource.
   */
  DataSource *Create(const std::string id);

 private:
  //! To avoid creating multiple instances of this factory in the program.
  DataSourceFactory() {}
  //! Map from DataSource identifier to creating function.
  std::map<const std::string, std::function<DataSource*(void)>> ds_map_;
};

}  // namespace lapis

#endif  // INCLUDE_DISK_DATA_SOURCE_H_
