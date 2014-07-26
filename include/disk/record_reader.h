// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-06-28 19:57

#ifndef INCLUDE_DISK_RECORD_READER_H_
#define INCLUDE_DISK_RECORD_READER_H_

#include <string>
#include <vector>
#include <map>
#include <functional>

#include "proto/model.pb.h"

namespace lapis {
/**
 * This is the interface for reading training data.
 * users have to implement this interface and register
 * their own record reader class with a name (e.g., in the main function).
 * We require labelled data should have a key e.g., filename of the image,
 * to identify the label and feature for the same record. I.e., the label
 * source and feature source should contain the key for the same the record.
 */
class RecordReader {
 public:
  /**
   * Initialization
   * The reader will locate one input file (e.g., an image) based on the prefix
   * and one element from suffix.
   * @param ds_proto provide meta info about the dataset, e.g., the path prefix
   * , i.e., the folder, where the images are located, or a full path for the
   * single label file.  If the path_prefix is a directory, then for each
   * path_suffix, they construct a full path to an input file (e.g., an image)
   * Besides, it also has shape/size info about the expected input data, e.g.,
   * the width and height of rgb image, based on which the reader may resize
   * the input image.
   * @param path_suffix, each suffix can be a full path or the last part of a
   * full path. For the imagenet dataset, the label source is in a single file,
   * hence the path_prefix is the full label file path, path_suffix is empty.
   * The format of each line of the label file is like: <image_filename,
   * labelid>. The feature source is a directory of images, hence the
   * path_prefix is the image directory, and the path_suffix is the
   * image_filename from the label source file.
   * @param offset the offset to the begin to a file if data is from a single
   * file; otherwise the offset to the first file in t he file list (e.g., the
   * list provided by path_suffix). if not =0, the reader is restored from some
   * checkpoint.
   */
  virtual void Init(const DataSourceProto &ds_proto,
                    const std::vector<std::string> &path_suffix,
                    int offset = 0) = 0;
  /**
   * Read next data record (e.g., one image or label).
   * @param key the identifier of the record.
   * @param val the feature of the record. the caller, i.e., the DataSource
   * knows the size of each record, hence it will allocte the mem for val.
   * @return true if not at the end of reading, otherwise return false
   */
  virtual bool ReadNextRecord(std::string *key, float *val) = 0;
  /**
   * Reset to read from the beginning.
   * Sometimes we need to go through the dataset multiple times.
   */
  virtual void Reset() = 0;
  /**
   * Return offset of the record to be read.
   * offset to the begin to a file if data is from a single file; otherwise
   * the offset to the first file in the file list (e.g., the list provided by
   * path_suffix). It will be checkpointed by the DataSource when doing
   * checkpoint.
   */
  virtual int Offset() = 0;
  /**
   * Return the identifier of this reader, will be used to set the id reader
   * filed in DataSourceProto.
   */
  virtual std::string id() = 0;
};

/*****************************************************************************/
/**
 * Register RecordReader with identifier ID
 * @param ID identifier of the reader e.g., "RGBFeature". The reader field in
 * DataSourceProto should be the same to this identifier
 * @param READER the child RecordReader
 */
#define REGISTER_READER(ID, READER) RecordReaderFactory::Instance()->\
  RegisterCreateFunction(ID, [](void)-> RecordReader* {return new READER();})

/**
 * Factory for creating record reader instance based on user provided reader
 * identifier.
 * Users are required to register user-defined readers before creating
 * instances of them during runtime through this factory. For example, if you
 * define a new record reader FooRecordReader with identifier "Foo", then you
 * can use it in your net by 1) configure your DataSrouceProto with the reader
 * field to be "Foo". 2) register it (e.g., at the start of the program). Then
 * your FooRecordReader will be created by calling
 * RecordReaderFactory::Instance()->Create("Foo") automatically by the
 * DataSource
 */

class RecordReaderFactory {
 public:
  /**
   * Static method to get instance of this factory, there should be only one
   * instance of this factory.
   */
  static RecordReaderFactory *Instance();

  /**
   * Register user defined reader, i.e., add the reader class with its
   * identifier (the reader field in DataSourceProto) into a inner map.
   * Later, the factory can then create an instance of the reader based on its
   * identifier. This function will be called by the REGISTER_READER macro.
   */
  void RegisterCreateFunction(
    const std::string id,
    std::function<RecordReader*(void)> create_function);
  /**
   * Create an instance the child RecordReader of identifier being id.
   * @param id the identifier of the child RecordReader.
   */
  RecordReader *Create(const std::string id);

 private:
  //! To avoid creating multiple instances of this factory in the program.
  RecordReaderFactory() {}
  //! Map from reader identifier to reader creating function.
  std::map<std::string, std::function<RecordReader*(void)>> reader_map_;
};

}  // namespace lapis

#endif  // INCLUDE_DISK_RECORD_READER_H_
