// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-06-28 19:57

#include "proto/lapis.pb.h"

namespace lapis {
template<class K, class V>
class DataReader {
 public:
  DataReader(const char* filename): filename_(filename) {}
  virtual static void parse(string line, std::pair<K, V>& record) = 0;

  template<class T>
  class RecordIterator :
      public std::iterator<std::forward_iterator_tag, class T> {
   public:
    RecordIterator(const char* filename);
    RecordIterator(const std::ifstream* fin): fin_(fin);

    T& operator*();

    void operator++();

    ~RecordIterator() {
      fin_->close();
    }
   private:
    std::ifstream *fin_;
    std::string line;
  };
  typedef RecordIterator<std::pair<K, V>> iterator;
  iterator& begin();
  iterator& end();

  ~DataReader() {}
 private:
  const char* filename_;
};
}  // namespace lapis
