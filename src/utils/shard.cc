// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-11-19 13:42

#include <sys/stat.h>
#include <glog/logging.h>

#include "utils/shard.h"

Shard::Shard(std::string folder, char mode, int capacity){
  struct stat sb;
  if(stat(folder.c_str(), &sb) == 0 && S_ISDIR(sb.st_mode)){
    LOG(INFO)<<"Open shard folder "<<folder;
  }else{
    LOG(FATAL)<<"Cannot open shard folder "<<folder;
  }

  path_= folder+"/shard.dat";
  if(mode==Shard::kRead){
    fdat_.open(path_, std::ios::in|std::ios::binary);
    CHECK(fdat_.is_open())<<"Cannot create file "<<path_;
  }
  if(mode==Shard::kCreate){
    fdat_.open(path_, std::ios::binary|std::ios::out|std::ios::trunc);
    CHECK(fdat_.is_open())<<"Cannot create file "<<path_;
  }
  if(mode==Shard::kAppend){
    int last_tuple=PrepareForAppend(path_);
    fdat_.open(path_, std::ios::binary|std::ios::out|std::ios::in|std::ios::ate);
    CHECK(fdat_.is_open())<<"Cannot create file "<<path_;
    fdat_.seekp(last_tuple);
  }

  mode_=mode;
  offset_=0;
  bufsize_=0;
  capacity_=capacity;
  buf_=new char[capacity];
}

Shard:: ~Shard(){
  delete buf_;
  fdat_.close();
}

bool Shard::Insert(const std::string& key, const Message& tuple) {
  std::string str;
  tuple.SerializeToString(&str);
  return Insert(key, str);
}
// insert one complete tuple
bool Shard::Insert(const std::string& key, const std::string& tuple) {
  if(keys_.find(key)!=keys_.end()||tuple.size()==0)
    return false;
  int size=key.size()+tuple.size()+2*sizeof(size_t);
  if(offset_+size>capacity_){
    fdat_.write(buf_, offset_);
    offset_=0;
    CHECK_LE(size, capacity_)<<"Tuple size is larger than capacity"
      <<"Try a larger capacity size";
  }
  *reinterpret_cast<size_t*>(buf_+offset_)=key.size();
  offset_+=sizeof(size_t);
  memcpy(buf_+offset_, key.data(), key.size());
  offset_+=key.size();
  *reinterpret_cast<size_t*>(buf_+offset_)=tuple.size();
  offset_+=sizeof(size_t);
  memcpy(buf_+offset_, tuple.data(), tuple.size());
  offset_+=tuple.size();
  return true;
}

void Shard::Flush() {
  fdat_.write(buf_, offset_);
  fdat_.flush();
  offset_=0;
}

int Shard::Next(std::string *key){
  key->clear();
  int ssize=sizeof(size_t);
  if(!PrepareNextField(ssize))
    return 0;
  CHECK_LE(offset_+ssize, bufsize_);
  int keylen=*reinterpret_cast<size_t*>(buf_+offset_);
  offset_+=ssize;

  if(!PrepareNextField(keylen))
    return 0;
  CHECK_LE(offset_+keylen, bufsize_);
  for(int i=0;i<keylen;i++)
    key->push_back(buf_[offset_+i]);
  offset_+=keylen;

  if(!PrepareNextField(ssize))
    return 0;
  CHECK_LE(offset_+ssize, bufsize_);
  int tuplelen=*reinterpret_cast<size_t*>(buf_+offset_);
  offset_+=ssize;

  if(!PrepareNextField(tuplelen))
    return 0;
  CHECK_LE(offset_+tuplelen, bufsize_);
  return tuplelen;
}

bool Shard::Next(std::string *key, Message* tuple) {
  int tuplelen=Next(key);
  if(tuplelen==0)
    return false;
  tuple->ParseFromArray(buf_+offset_, tuplelen);
  offset_+=tuplelen;
  return true;
}


bool Shard::Next(std::string *key, std::string* tuple) {
  int tuplelen=Next(key);
  if(tuplelen==0)
    return false;
  tuple->clear();
  for(int i=0;i<tuplelen;i++)
    tuple->push_back(buf_[offset_+i]);
  offset_+=tuplelen;
  return true;
}

void Shard::SeekToFirst(){
  CHECK_EQ(mode_, kRead);
  bufsize_=0;
  offset_=0;
  fdat_.close();
  fdat_.open(path_, std::ios::in|std::ios::binary);
  CHECK(fdat_.is_open())<<"Cannot create file "<<path_;
}

// if the buf does not have the next complete field, read data from disk
bool Shard::PrepareNextField(int size){
  if(offset_+size>bufsize_){
    bufsize_-=offset_;
    CHECK_LE(bufsize_, offset_);
    for(int i=0;i<bufsize_;i++)
      buf_[i]=buf_[i+offset_];
    offset_=0;
    if(fdat_.eof())
      return false;
    else{
      fdat_.read(buf_+bufsize_, capacity_-bufsize_);
      bufsize_+=fdat_.gcount();
    }
  }
  return true;
}


const int Shard::Count() {
  std::ifstream fin(path_, std::ios::in|std::ios::binary);
  CHECK(fdat_.is_open())<<"Cannot create file "<<path_;
  int count=0;
  while(true){
    size_t len;
    fin.read(reinterpret_cast<char*>(&len), sizeof(len));
    if(fin.good())
      fin.seekg(len, std::ios_base::cur);
    else break;
    if(fin.good())
      fin.read(reinterpret_cast<char*>(&len), sizeof(len));
    else break;
    if(fin.good())
      fin.seekg(len, std::ios_base::cur);
    else break;
    if(!fin.good())
      break;
    count++;
  }
  fin.close();
  return count;
}

int Shard::PrepareForAppend(std::string path){
  std::ifstream fin(path, std::ios::in|std::ios::binary);
  if(!fin.is_open()){
    fdat_.open(path, std::ios::out|std::ios::binary);
    fdat_.flush();
    fdat_.close();
    return 0;
  }

  int last_tuple_offset=0;
  char buf[256];
  size_t len;
  while(true){
    memset(buf, 0, 256);
    fin.read(reinterpret_cast<char*>(&len), sizeof(len));
    if(fin.good())
      fin.read(buf, len);
    else break;
    if(fin.good())
      fin.read(reinterpret_cast<char*>(&len), sizeof(len));
    else break;
    if(fin.good())
      fin.seekg(len, std::ios_base::cur);
    else break;
    if(fin.good())
      keys_.insert(std::string(buf));
    else break;
    last_tuple_offset=fin.tellg();
  }
  fin.close();
  return last_tuple_offset;
}
