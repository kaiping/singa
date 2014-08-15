// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-06-28 14:41

#include "utils/proto_helper.h"
#include <fcntl.h>

#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <glog/logging.h>

#include <fstream> //NOLINT

using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using std::fstream;

namespace lapis {
template<K, V, M>
std::map<K, V> ToStdMap(const M& gmap) {
  std::map<K, V> stdmap;
  for(auto& pair: gmap)
    stdmap[pair.first]=pair.second;
  return stdmap;
}
template<K, V, M, P>
M* ToGoogleMap(std::map<K,V> stdmap){
  M* gmap=new M();
  for(auto& entry: stdmap) {
    P *pair=gmap.add_pair();
    pair->set_key(entry.first);
    pair->set_val(entry.second);
  }
  return gmap;
}

void ReadProtoFromTextFile(const char *filename, Message *proto) {
  int fd = open(filename, O_RDONLY);
  CHECK_NE(fd, -1) << "File not found: " << filename;
  FileInputStream *input = new FileInputStream(fd);
  CHECK(google::protobuf::TextFormat::Parse(input, proto));
  delete input;
  close(fd);
}

void WriteProtoToTextFile(const Message &proto, const char *filename) {
  int fd = open(filename, O_WRONLY | O_CREAT);
  CHECK_NE(fd, -1) << "File not created: " << filename;
  FileOutputStream *output = new FileOutputStream(fd);
  CHECK(google::protobuf::TextFormat::Print(proto, output));
  delete output;
  close(fd);
}

void ReadProtoFromBinaryFile(const char *filename, Message *proto) {
  fstream input(filename, fstream::in | fstream::binary);
  CHECK(proto->ParseFromIstream(&input));
}

void WriteProtoToBinaryFile(const Message &proto, const char *filename) {
  fstream output(filename, fstream::out | fstream::trunc | fstream::binary);
  CHECK(proto.SerializeToOstream(&output));
}

}  // namespace lapis
