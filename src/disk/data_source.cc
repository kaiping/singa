// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-16 22:01

#include "model/data_source.h"
namespace lapis {
/*****************************************************************************
 * Implementation for DataSource
 ****************************************************************************/
DataSource::DataSource(const DataSourceProto &ds_proto) {
}

void DataSource::GetData(Blob *blob) {
}

void DataSource::ToProto(DataSourceProto *ds_proto) {
  ds_proto->set_name(name_);
  ds_proto->set_parser(parser_);
  ds_proto->set_path(path_);
  ds_proto->set_size(size_);
  ds_proto->set_type(type_);

  ds_proto->set_channels(channels_);
  ds_proto->set_height(height_);
  ds_proto->set_width(width_);
}

}  // namespace lapis
