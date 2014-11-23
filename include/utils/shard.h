// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-11-19 12:14
#ifndef DATASOURCE_SHARD_H_
#define DATASOURCE_SHARD_H_

#include <google/protobuf/message.h>
#include <fstream>
#include <string>
#include <unordered_set>


using google::protobuf::Message;

/**
 * Every node has a shard storing training records/tuples;
 * There are two files, one is shard.key storing keys of all tuples to avoid
 * redundant tuples; the other file is shard.dat storing the tuple data, in the
 * form of [key_len key tuple_len tuple] (key_len and tuple_len are of type
 * uint32, which indicate the bytes of key and tuple respectively.
 *
 * Every insert writes the shard.key firstly; When Shard obj
 * is created, it will remove the last key if the tuple size and key size do
 * not match because the last write of tuple crashed.
 *
 */
class Shard {
 public:
  // read only mode used in training
  static const char kRead=0;
  // write mode used in creating shard (will overwrite previous one)
  static const char kCreate=1;
  // append mode, e.g. used when previous creating crashes
  static const char kAppend=2;

 public:
  /*
   * init the shard obj
   * @folder shard folder on worker node, need large disk space
   * @mode either kRead, kWrite or kAppend
   * @bufsize batch bufsize bytes data for every disk op (read or write),
   * default is 32MB
   */
  Shard(std::string folder, char mode, int capacity=33554432);
  ~Shard();

  // read next tuple
  bool Next(std::string *key, Message* tuple);
  bool Next(std::string *key, std::string* tuple);

  // insert tuple; if inserted before, then return false
  bool Insert(const std::string& key, const Message& tuple);
  bool Insert(const std::string& key, const std::string& tuple);

  // valid only for kRead mode
  void SeekToFirst();
  // valid only for kRead mode
  void Flush() ;
  /**
   * return num of tuples
   */
  const int Count();

 protected:
  int Next(std::string *key);
  int PrepareForAppend(std::string path);
  bool PrepareNextField(int size);

 private:
  char mode_;
  std::string path_;
  // either ifstream or ofstream
  std::fstream fdat_;
  // to avoid replicated tuples
  std::unordered_set<std::string> keys_;
  // internal buffer
  char* buf_;
  // offset inside the buf_
  int offset_;
  // allocated bytes for the buf_
  int capacity_;
  // bytes in buf_, used in reading
  int bufsize_;
};
#endif  // DATASOURCE_SHARD_H_
