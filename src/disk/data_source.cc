// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-16 22:01
#include <glog/logging.h>

#include "disk/data_source.h"
namespace lapis {
/*****************************************************************************
 * Implementation for DataSource
 ****************************************************************************/
void DataSource::Init(const DataSourceProto &ds_proto) {
  size_=ds_proto.size();
  name_=ds_proto.name();
  offset_=ds_proto.offset();
  //record_size_=channels_*height_*width_*sizeof(float);
}

void DataSource::ToProto(DataSourceProto *ds_proto) {
  ds_proto->set_size(size_);
  ds_proto->set_name(name_);
  ds_proto->set_offset(offset_);
  ds_proto->set_id(id());
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
