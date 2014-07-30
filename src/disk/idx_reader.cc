// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-20 16:26

#include <glog/logging.h>
#include <arpa/inet.h>
#include <cstdint>

#include "disk/idx_reader.h"

namespace lapis {
/******************************************************************************
 * Conversion between Big Endian and Little Endian
 * The following implementation is provided by John Dibling from
 * <a href="http://stackoverflow.com/a/17448956"> StackOverflow</a>
 *****************************************************************************/
template<class Val> inline Val ntohx(const Val &in) {
  char out[sizeof(in)] = {0};
  for ( size_t i = 0; i < sizeof(Val); ++i )
    out[i] = (reinterpret_cast<const char *>(&in))[sizeof(Val) - i - 1];
  return *(reinterpret_cast<Val *>(out));
}

template<> inline unsigned char ntohx<unsigned char>(const unsigned char
    &v) { //NOLINT
  return v;
}
template<> inline uint16_t ntohx<uint16_t>(const uint16_t &v) {
  return ntohs(v);
}

template<> inline uint32_t ntohx<uint32_t>(const uint32_t &v) {
  return ntohl(v);
}

template<> inline uint64_t ntohx<uint64_t>(const uint64_t &v) {
  uint32_t ret[] = {
    ntohl(((const uint32_t *)&v)[1]),
    ntohl(((const uint32_t *)&v)[0])
  };
  return *(reinterpret_cast<uint64_t *>(&ret[0]));
}
template<> inline float ntohx<float>(const float &v) {
  uint32_t const *cast = reinterpret_cast<uint32_t const *>(&v);
  uint32_t ret = ntohx(*cast);
  return *(reinterpret_cast<float *>(&ret));
}

template<class Val> inline Val htonx(const Val &in) {
  char out[sizeof(in)] = {0};
  for ( size_t i = 0; i < sizeof(Val); ++i )
    out[i] = (reinterpret_cast<char *>(&in))[sizeof(Val) - i - 1];
  return *(reinterpret_cast<Val *>(out));
}

template<> inline unsigned char htonx<unsigned char>(const unsigned char
    &v) { //NOLINT
  return v;
}
template<> inline uint16_t htonx<uint16_t>(const uint16_t &v) {
  return htons(v);
}

template<> inline uint32_t htonx<uint32_t>(const uint32_t &v) {
  return htonl(v);
}

template<> inline uint64_t htonx<uint64_t>(const uint64_t &v) {
  uint32_t ret[] = {
    htonl(((const uint32_t *)&v)[1]),
    htonl(((const uint32_t *)&v)[0])
  };
  return *(reinterpret_cast<uint64_t *>(&ret[0]));
}
template<> inline float htonx<float>(const float &v) {
  uint32_t const *cast = reinterpret_cast<uint32_t const *>(&v);
  uint32_t ret = htonx(*cast);
  return *(reinterpret_cast<float *>(&ret));
}


/*****************************************************************************
 * Implementation for IDXReader
 ****************************************************************************/

const std::string IDXReader::id_ = "IDXReader";
void IDXReader::Init(const DataSourceProto &ds_proto,
                     const std::vector<std::string> &suffix,
                     int offset) {
  is_.open(ds_proto.path(), std::ofstream::in | std::ofstream::binary);
  CHECK(is_.is_open()) << "Error open the label file "
                       << ds_proto.path() << "\n";
  if (offset > 0) {
    is_.seekg(0, is_.end);
    int size = is_.tellg();
    CHECK(offset < size) << "the offset " << offset
                         << " should be < the file size " << size << "\n";
    is_.seekg(offset, is_.beg);
  }
  char magic[4] = {0};
  is_.read(magic, 4);
  type_ = static_cast<int>(magic[2]);
  int dim = static_cast<int>(magic[3]);
  CHECK_LE(dim, 3) << "Feature dimension is " << dim
                   << " in file " << ds_proto.path() << "which should <= 3\n";
  LOG(INFO) << "type is " << type_ << " dim is " << dim << "\n";
  int num_records = 0;
  is_.read(reinterpret_cast<char *>(&num_records), 4);
  num_records = ntohx(num_records);
  LOG(INFO) << num_records << " instances\n";
  CHECK_EQ(num_records, ds_proto.size());
  length_ = 1;
  for (int i = 1, d; i < dim; i++) {
    is_.read(reinterpret_cast<char *>(&d), 4);
    d = ntohx(d);
    LOG(INFO) << "one dim is " << d << "\n";
    length_ *= d;
  }
  int ds_length = ds_proto.width() * ds_proto.height() * ds_proto.channels();
  CHECK_EQ(length_, ds_length)
      << "The length of one record from the idx file is "
      << length_ << " , while that from the DataSourceProto is " << ds_length;
  switch (type_) {
  case kIDX_ubyte: {
    num_bytes_ = length_;
    break;
  }
  case kIDX_byte: {
    num_bytes_ = length_;
    break;
  }
  // case kIDX_short: { num_bytes_=length_*sizeof(short); break;}
  case kIDX_float: {
    num_bytes_ = length_ * sizeof(float);
    break;
  }
  // case kIDX_double: { num_bytes_=length_*sizeof(double); break;}
  default: {
    LOG(ERROR) << "Data type " << type_ << " is not supported for the IDX file "
               << ds_proto.path() << "\n"; break;
  }
  }
  buf_ = new char[num_bytes_];
}

bool IDXReader::ReadNextRecord(std::string *key, float *val) {
  if (is_.eof())
    return false;
  is_.read(buf_, num_bytes_);
  switch (type_) {
  case kIDX_ubyte: {
    unsigned char x;
    for (int i = 0; i < length_; i++) {
      x = static_cast<unsigned char>(buf_[i]);
      val[i] = static_cast<float>(x);
    }
    break;
  }
  case kIDX_float: {
    for (int i = 0; i < length_; i++) {
      val[i] = *(reinterpret_cast<float *>(buf_ + i * sizeof(float)));
      val[i] = ntohx(val[i]);
    }
    break;
  }
  default: {
    LOG(ERROR) << "The data type is not supported\n";
    break;
  }
  }
  return true;
}

void IDXReader::Reset() {
  is_.seekg(0, is_.beg);
}

int IDXReader::Offset() {
  return is_.tellg();
}

IDXReader::~IDXReader() {
  delete buf_;
  is_.close();
}

}  // namespace lapis

