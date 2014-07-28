// Copyright © 2014 Anh Dinh. All Rights Reserved.
// piccolo.h

#ifndef INCLUDE_CORE_FILE_H_
#define INCLUDE_CORE_FILE_H_

#include "boost/noncopyable.hpp"
#include "core/common.h"
#include "proto/common.pb.h"

#include <stdio.h>
#include <glob.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>


namespace google {
namespace protobuf {
class Message;
}
}

namespace lapis {

class File {
 public:
  virtual ~File() {}
  virtual int read(char *buffer, int len) = 0;
  virtual bool read_line(string *out) = 0;
  virtual bool eof() = 0;
  virtual void seek(int64_t pos) = 0;
  virtual uint64_t tell() = 0;
  virtual const char *name() {
    return "";
  }
  virtual void sync() = 0;

  int write_string(const string &buffer) {
    return write(buffer.data(), buffer.size());
  }

  virtual int write(const char *buffer, int len) = 0;

  string readLine() {
    string out;
    read_line(&out);
    return out;
  }

  struct Info {
    string name;
    struct stat stat;
  };

  static string Slurp(const string &file);
  static void Dump(const string &file, StringPiece data);
  static void Mkdirs(string path);
  static vector<string> MatchingFilenames(StringPiece glob);
  static vector<Info> MatchingFileinfo(StringPiece glob);

  static bool Exists(const string &path);
  static void Move(const string &src, const string &dst);

 private:
};

class LocalFile : public File {
 public:
  LocalFile(FILE *fp);
  LocalFile(const string &path, const string &mode);
  virtual ~LocalFile() {
    if (close_on_delete) {
      fflush(fp);
      fclose(fp);
    }
  }

  void sync() {
    fsync(fileno(fp));
  }

  bool read_line(string *out);
  int read(char *buffer, int len);
  int write(const char *buffer, int len);
  void seek(int64_t pos) {
    fseek(fp, pos, SEEK_SET);
  }
  uint64_t tell() {
    return ftell(fp);
  }

  void Printf(const char *p, ...);
  virtual FILE *filePointer() {
    return fp;
  }

  const char *name() {
    return path.c_str();
  }

  bool eof();

 private:
  FILE *fp;
  string path;
  bool close_on_delete;
};

class Encoder {
 public:
  Encoder(string *s) : out_(s), out_f_(NULL) {}
  Encoder(File *f) : out_(NULL), out_f_(f) {}

  template <class T>
  void write(const T &v);

  void write_string(StringPiece v);
  void write_bytes(StringPiece s);
  void write_bytes(const char *a, int len);

  string *data() {
    return out_;
  }

  size_t pos() {
    if (out_) {
      return out_->size();
    }
    return out_f_->tell();
  }

 private:
  string *out_;
  File *out_f_;
};

class Decoder {
 private:
  const string *src_;
  const char *src_p_;
  File *f_src_;

  size_t pos_;
 public:
  Decoder(const string &data) : src_(&data), src_p_(data.data()), f_src_(NULL),
    pos_(0) {}
  Decoder(File *f) : src_(NULL), src_p_(NULL), f_src_(f), pos_(0) {}

  template <class V> void read(V *t) {
    if (src_) {
      memcpy(t, src_p_ + pos_, sizeof(V));
      pos_ += sizeof(V);
    } else {
      f_src_->read((char *)t, sizeof(t));
    }
  }

  template <class V> V read() {
    V v;
    read<V>(&v);
    return v;
  }

  void read_bytes(char *b, int len) {
    if (src_p_) {
      memcpy(b, src_p_ + pos_, len);
    } else {
      f_src_->read(b, len);
    }
    pos_ += len;
  }

  void read_string(string *v) {
    uint32_t len;
    read<uint32_t>(&len);
    v->resize(len);
    read_bytes(&(*v)[0], len);
  }

  bool done() {
    if (src_) {
      return pos_ == src_->size();
    } else {
      return f_src_->eof();
    }
  }

  size_t pos() {
    if (src_) {
      return pos_;
    }
    return f_src_->tell();
  }

  void seek(int p) {
    if (src_) {
      pos_ = p;
    } else {
      pos_ = p;
      f_src_->seek(pos_);
    }
  }
};

class RecordFile {
 public:
  enum CompressionType {
    NONE = 0,
    LZO = 1
  };

  RecordFile() : fp(NULL) {}

  RecordFile(const string &path, const string &mode, int compression = NONE);
  virtual ~RecordFile();

  virtual void write(const google::protobuf::Message &m);
  virtual bool read(google::protobuf::Message *m);

  const char *name() {
    return fp->name();
  }

  bool eof() {
    return fp->eof();
  }
  void sync() {
    fp->sync();
  }

  void seek(uint64_t pos);

  void writeChunk(StringPiece data);
  bool readChunk(string *s);

  File *fp;

 private:
  string buf_;
  string decomp_buf_;
  string decomp_scratch_;
  string path_;
  string mode_;
};
}  // namespace lapis

#endif  // INCLUDE_CORE_FILE_H_
