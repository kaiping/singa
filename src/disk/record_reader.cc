// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-18 22:17

#include "disk/record_reader.h"
/*****************************************************************************
 * Implementation for RecordReaderFactory
 ****************************************************************************/
RecordReaderFactory *RecordReaderFactory::Instance() {
  static RecordReaderFactory* factory;
  return &factory;
}
void RecordReaderFactory::RegisterCreateFunction(
    const std::string id,
    std::function<RecordReader*(void)> create_function) {
  reader_map_[id]=create_function;
}

RecordReader *RecordReaderFactory::Create(const std::string id) {
  CHECK(reader_map_.find(id)!=reader_map_.end())<<"The reader "<<id
                                                <<" has not been registered\n";
  return layer_map_.at(id);
}
