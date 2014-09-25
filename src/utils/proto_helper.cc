// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-06-28 14:41

#include <stdint.h>
#include <fcntl.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <glog/logging.h>

#include <fstream> //NOLINT
#include <string>


#include "proto/common.pb.h"
#include "utils/stringpiece.h"
#include "utils/proto_helper.h"

using std::fstream;
using std::ios;
using std::max;
using std::string;
using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::ZeroCopyOutputStream;
using google::protobuf::io::CodedOutputStream;

namespace lapis {
const int kBufLen=1024;
void ReadProtoFromTextFile(const char* filename,
    ::google::protobuf::Message* proto) {
  int fd = open(filename, O_RDONLY);
  CHECK_NE(fd, -1) << "File not found: " << filename;
  FileInputStream* input = new FileInputStream(fd);
  CHECK(google::protobuf::TextFormat::Parse(input, proto));
  delete input;
  close(fd);
}
void WriteProtoToTextFile(const Message& proto, const char* filename) {
  int fd = open(filename, O_WRONLY);
  FileOutputStream* output = new FileOutputStream(fd);
  CHECK(google::protobuf::TextFormat::Print(proto, output));
  delete output;
  close(fd);
}
void ReadProtoFromBinaryFile(const char* filename, Message* proto) {
  VLOG(3)<<"read from binry file";
  int fd = open(filename, O_RDONLY);
  CHECK_NE(fd, -1) << "File not found: " << filename;
  ZeroCopyInputStream* raw_input = new FileInputStream(fd);
  CodedInputStream* coded_input = new CodedInputStream(raw_input);
  coded_input->SetTotalBytesLimit(536870912, 268435456);
  VLOG(3)<<"before parse";
  CHECK(proto->ParseFromCodedStream(coded_input));
  delete coded_input;
  delete raw_input;
  close(fd);
  VLOG(3)<<"read binry file";
}
void WriteProtoToBinaryFile(const Message& proto, const char* filename) {
  fstream output(filename, ios::out | ios::trunc | ios::binary);
  CHECK(proto.SerializeToOstream(&output));
}
/*
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
*/
const std::map<std::string, int> ToStdMap(const StringIntMap& gmap) {
  std::map<std::string, int> stdmap;
  for(auto& pair: gmap.pair())
    stdmap[pair.key()]=pair.val();
  return stdmap;
}


const std::map<int, int> ToStdMap(const IntIntMap& gmap) {
  std::map<int, int> stdmap;
  for(auto& pair: gmap.pair())
    stdmap[pair.key()]=pair.val();
  return stdmap;
}

const StringIntMap ToProtoMap(std::map<std::string,int> stdmap){
  StringIntMap gmap;
  for(auto& entry: stdmap) {
    StringIntPair *pair=gmap.add_pair();
    pair->set_key(entry.first);
    pair->set_val(entry.second);
  }
  return gmap;
}

const IntIntMap ToProtoMap(std::map<int, int> stdmap){
  IntIntMap gmap;
  for(auto& entry: stdmap) {
    IntIntPair *pair=gmap.add_pair();
    pair->set_key(entry.first);
    pair->set_val(entry.second);
  }
  return gmap;
}

const std::string FormatPerformance(int src, const Performance &perf) {
  int step=-1;
  float loss=-1.0f, precision=-1.0f;
  std::string prefix="unknown";
  if(perf.has_prefix())
    prefix=perf.prefix();
  if(perf.has_step())
    step=perf.step();
  if(perf.has_loss())
    loss=perf.loss();
  if(perf.has_precision())
    precision=perf.precision();
  char buf[kBufLen];
  sprintf(buf, "%10s, slot:%3d, step:%5d, loss:%6.4f, precision:%6.4f",
      prefix.c_str(), src, step, loss, precision);
  std::string s(buf);
  return s;
}

const std::string FormatTime(int step, double tcomp, double tcomm){
  char buf[kBufLen];
  sprintf(buf, "%4d, comp:%4.2f, comm:%4.2f", step, tcomp/step, tcomm/step);
  std::string s(buf);
  return s;
}
}  // namespace lapis
