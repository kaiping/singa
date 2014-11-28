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
 * Data shard stores training/validation/test records.
 * Every worker node should have a training shard (validation/test shard
 * is optional). The shard file for training is
 * lapis::Cluster::data_folder()/train/shard.dat; The shard file for validation
 * is lapis::Cluster::data_folder()/train/shard.dat; Similar path for test.
 *
 * shard.dat consists of a set of unordered records/tuples. The tuple is
 * encoded as [key_len key tuple_len tuple] (key_len and tuple_len are of type
 * uint32, which indicate the bytes of key and tuple respectively.
 *
 * When Shard obj is created, it will remove the last key if the tuple size and
 * key size do not match because the last write of tuple crashed.
 *
 */
class Shard {
 public:
  //!< read only mode used in training
  static const char kRead=0;
  //!< write mode used in creating shard (will overwrite previous one)
  static const char kCreate=1;
  //!< append mode, e.g. used when previous creating crashes
  static const char kAppend=2;

 public:
  /**
   * Init the shard obj.
   * @folder shard folder (path except shard.dat) on worker node
   * @mode shard open mode, Shard::kRead, Shard::kWrite or Shard::kAppend
   * @bufsize batch bufsize bytes data for every disk op (read or write),
   * default is 32MB
   */
  Shard(std::string folder, char mode, int capacity=33554432);
  ~Shard();

  /**
   * read next tuple from the shard.
   * @key key tuple key
   * @param val tuple value of type Message
   * @return true if read success otherwise false, e.g., the tuple was not
   * inserted completely.
   */
  bool Next(std::string *key, Message* val);
  /**
   * read next tuple from the shard.
   * @key key tuple key
   * @param val tuple value of type string
   * @return true if read success otherwise false, e.g., the tuple was not
   * inserted completely.
   */
  bool Next(std::string *key, std::string* val);

  /**
   * Append one tuple to the shard.
   * @param key tuple string, e.g., image path
   * @param val
   * @return reture if sucess, otherwise false, e.g., inserted before
   */
  bool Insert(const std::string& key, const Message& tuple);
  /**
   * Append one tuple to the shard.
   * @param key tuple string, e.g., image path
   * @param val
   * @return reture if sucess, otherwise false, e.g., inserted before
   */
  bool Insert(const std::string& key, const std::string& tuple);

  /**
   * Move the read pointer to the head of the shard file.
   * Used for repeated reading.
   */
  void SeekToFirst();
  /**
   * Flush buffered data to disk.
   * Used only for kCreate or kAppend.
   */
  void Flush() ;
  /**
   * @return num of tuples
   */
  const int Count();

 protected:
  /**
   * Read the next key and prepare buffer for reading value.
   * @param key
   * @return length (i.e., bytes) of value field.
   */
  int Next(std::string *key);
  /**
   * Setup the disk pointer to the right position for append in case that
   * the pervious write crashes.
   * @param path shard path.
   * @return offset (end pos) of the last success written tuple.
   */
  int PrepareForAppend(std::string path);
  /**
   * Read data from disk if the current data in the buffer is not a full field.
   * @param size size of the next field.
   */
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
