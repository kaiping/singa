// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-06-28 14:57
#ifndef INCLUDE_UTILS_PROTO_HELPER_H_
#define INCLUDE_UTILS_PROTO_HELPER_H_

#include <google/protobuf/message.h>
#include <map>
#include "proto/common.pb.h"
#include "proto/model.pb.h"


using google::protobuf::Message;

namespace lapis {

const IntIntMap ToProtoMap(std::map<int, int> stdmap);
const StringIntMap ToProtoMap(std::map<std::string,int> stdmap);
const std::map<int, int> ToStdMap(const IntIntMap& gmap);
const std::map<std::string, int> ToStdMap(const StringIntMap& gmap);
void ReadProtoFromTextFile(const char *filename, Message *proto);
void WriteProtoToTextFile(const Message &proto, const char *filename);
void ReadProtoFromBinaryFile(const char *filename, Message *proto);
void WriteProtoToBinaryFile(const Message &proto, const char *filename);
const std::string FormatPerformance(int src, const Performance& perf);
const std::string FormatTime(int step, double tcomp, double tcomm);
}  // namespace lapis

#endif  // INCLUDE_UTILS_PROTO_HELPER_H_

