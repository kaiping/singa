// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-16 22:01
#include <glog/logging.h>
#include <memory>

#include "datasource/label_source.h"
#include "datasource/rgb_dir_source.h"
#include "datasource/data_source.h"

namespace lapis {
/*****************************************************************************
 * Implementation for DataSource
 ****************************************************************************/
std::map<string, Shape> DataSource::ShapesOf(const DataSourceProtos &sources) {
  std::map<string, Shape> shape_map;
  for(auto& source: sources) {
    shape_map[source.name()]=source.shape();
  }
  return shape_map;
}

const std::shared_ptr<StringVec> DataSource::Init(
    const DataSourceProto &ds_proto,
    std::shared_ptr<StringVec>& filenames){
  size_ = ds_proto.shape().num();
  name_ = ds_proto.name();
  offset_ = ds_proto.offset();
  return filenames;
  // record_size_=channels_*height_*width_*sizeof(float);
}

void DataSource::ToProto(DataSourceProto *proto) {
  proto->set_offset(offset_);
}

/*
void DataSource::GetData(Blob *blob) {
  string key;
  float *addr=blob->content();
  for (int i=0;i<blob->num();i++)
    reader_->ReadNextRecord(&key, addr+i*record_size_);
}
*/
/*****************************************************************************
 * Implementation of DataSourceFactory
 ****************************************************************************/
#define CreateDS(DSClass) [](void)->DataSource* {return new DSClass();}

std::shared_ptr<DataSourceFactory> DataSourceFactory::instance_;

std::shared_ptr<DataSourceFactory> DataSourceFactory::Instance() {
   if (!instance_.get())
     instance_.reset(new DataSourceFactory());
   return instance_;
}

DataSourceFactory::DataSourceFactory() {
  RegisterCreateFunction(LabelSource::type, CreateDS(LabelSource));
  RegisterCreateFunction(RGBDirSource::type, CreateDS(RGBDirSource));
}

void DataSourceFactory::RegisterCreateFunction(
  const string &id,
  std::function<DataSource*(void)> create_function) {
  ds_map_[id] = create_function;
  DLOG(INFO)<<"register DataSource: "<<id;
}

DataSource *DataSourceFactory::Create(const string id) {
  CHECK(ds_map_.find(id) != ds_map_.end()) << "The reader " << id
      << " has not been registered\n";
  return ds_map_.at(id)();
}
}  // namespace lapis
