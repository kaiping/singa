#include <fcntl.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include "utils/common.h"
using std::fstream;
using std::ios;
using std::max;
using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::ZeroCopyOutputStream;
using google::protobuf::io::CodedOutputStream;

namespace singa {

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
  int fd = open(filename, O_WRONLY | O_CREAT, 0644);
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
/*
void WriteProtoToBinaryFile(const Message& proto, const char* filename) {
  fstream output(filename, ios::out | ios::trunc | ios::binary);
  CHECK(proto.SerializeToOstream(&output));
}
*/

/**
 * Formatted string.
 */
string VStringPrintf(string fmt, va_list l) {
  char buffer[32768];
  vsnprintf(buffer, 32768, fmt.c_str(), l);
  return string(buffer);
}

/**
 * Formatted string.
 */
string StringPrintf(string fmt, ...) {
  va_list l;
  va_start(l, fmt); //fmt.AsString().c_str());
  string result = VStringPrintf(fmt, l);
  va_end(l);
  return result;
}

void Debug() {
  int i = 0;
  char hostname[256];
  gethostname(hostname, sizeof(hostname));
  printf("PID %d on %s ready for attach\n", getpid(), hostname);
  fflush(stdout);
  while (0 == i)
    sleep(5);
}
}  // namespace singa
