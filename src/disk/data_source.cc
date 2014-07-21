// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-16 22:01

#include "disk/data_source.h"
#include "disk/record_reader.h"
namespace lapis {
/*****************************************************************************
 * Implementation for DataSource
 ****************************************************************************/
DataSource::DataSource(const DataSourceProto &ds_proto) {
  reader_=RecordReaderFactory::Instance()->Create(ds_proto->reader());
  reader_.Init(ds_proto);

  size_=ds_proto.size();
  name_=ds_proto.name();
  path_=ds_proto.path();
  channels_=ds_proto.channels();
  height_=ds_proto.height();
  width_=ds_proto.width();

  record_size_=channels_*height_*width_*sizeof(float);
}

DataSource::~DataSource() {
  if (reader_!=nullptr)
    delete reader;
}

void DataSource::GetData(Blob *blob) {
  std::string key;
  float *addr=blob->content();
  for (int i=0;i<blob->num();i++)
    reader_->ReadNextRecord(&key, addr+i*record_size_);
}

void DataSource::ToProto(DataSourceProto *ds_proto) {
  ds_proto->set_name(name_);
  ds_proto->set_path(path_);
  ds_proto->set_size(size_);
  // ds_proto->set_type(type_);

  ds_proto->set_channels(channels_);
  ds_proto->set_height(height_);
  ds_proto->set_width(width_);
  ds_proto->set_reader(reader->id());
}

}  // namespace lapis
