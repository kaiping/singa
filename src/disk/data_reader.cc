// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-06-28 21:26

#include "disk/data_reader.h"


namespace lapis {
DataReader::iterator& DataReader::begin() {
  DataReader::iterator* iter = new DataReader::RecordIterator(filename_);
  return *iter
}

DataReader::iterator& end() {
  auto* fin = new std::ifstream(filename_);
  fin.setstate(std::ios::eofbit);
  auto* iter = new DataReader::RecordIterator(fin);
  return *iter
}

DataReader::RecordIterator::RecordIterator(const char* filename) {
  fin_ = new std::ifstream(filename);
}

DataReader::iterator& DataReader::RecordIterator::operator++() {
  if (fin_->eof())
    return *this;
  std::getline(fin_, line);
  return *this;
}

template<K, V>
T& DataReader::RecordIterator::operator*() {
  T* t = new T();
  parse(line_, t);
  return *t;
}

bool DataReader::RecordIterator::operator!=(
    const DataReader::RecordIterator& rhs) {
  return fin_ != rhs.fin_;
}
}  // namespace lapis
