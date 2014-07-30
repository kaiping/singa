// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-16 22:01
#include <glog/logging.h>

#include "disk/data_source.h"
namespace lapis {
/*****************************************************************************
 * Implementation for DataSource
 ****************************************************************************/
void DataSource::Init(const DataSourceProto &proto) {
  size_ = proto.size();
  name_ = proto.name();
  offset_ = proto.offset();
  // record_size_=channels_*height_*width_*sizeof(float);
}

void DataSource::ToProto(DataSourceProto *proto) {
  proto->set_size(size_);
  proto->set_name(name_);
  proto->set_offset(offset_);
  proto->set_id(id());
}


/*
void DataSource::GetData(Blob *blob) {
  std::string key;
  float *addr=blob->content();
  for (int i=0;i<blob->num();i++)
    reader_->ReadNextRecord(&key, addr+i*record_size_);
}
*/
/*****************************************************************************
 * Implementation of DataSourceFactory
 ****************************************************************************/
DataSourceFactory *DataSourceFactory::Instance() {
  /**
   * using shared_ptr
   * static std::shared_ptr<DataSourceFactory> factory;
   * if (!factory.Get())
   *   factory.reset(new DataSourceFactory());
   * return factory;
   */
  static DataSourceFactory factory;
  return &factory;
}
void DataSourceFactory::RegisterCreateFunction(
  const std::string &id,
  std::function<DataSource*(void)> create_function) {
  ds_map_[id] = create_function;
}

DataSource *DataSourceFactory::Create(const std::string id) {
  CHECK(ds_map_.find(id) != ds_map_.end()) << "The reader " << id
      << " has not been registered\n";
  return ds_map_.at(id)();
}
}  // namespace lapis
