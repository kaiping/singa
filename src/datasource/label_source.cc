// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-02 19:50

#include <glog/logging.h>
#include <algorithm>
#include <random>
#include <chrono>

#include "datasource/label_source.h"
using std::shared_ptr;
namespace lapis {
const std::string LabelSource::type = "LabelSource";
const shared_ptr<StringVec> LabelSource::Init(const DataSourceProto &ds_proto,
      shared_ptr<StringVec>& filenames){
  VLOG(3)<<"init label source";
  DataSource::Init(ds_proto,filenames);
  label_path_ = ds_proto.path();
  return LoadData(filenames);
}

void LabelSource::ToProto(DataSourceProto *proto) {
  proto->set_path(label_path_);
}

void LabelSource::GetData(Blob *blob) {
  float *addr = blob->dptr;
  for (int i = 0; i < blob->num(); i++) {
    if (offset_ == size_)
      offset_ = 0;
    addr[i] = labels_[offset_++];
  }
}
void LabelSource::NextRecord(FloatVector *record){
  record->set_data(0, labels_[offset_]);
  offset_++;
}


const shared_ptr<StringVec> LabelSource::LoadData(
  const shared_ptr<StringVec> &keys) {
  DLOG(INFO)<<"Load Label Data...";
  std::ifstream is(label_path_);
  CHECK(is.is_open()) << "Error open the label file " << label_path_;
  int v;
  std::string k;
  image_names_ = std::make_shared<StringVec>();
  std::vector<std::pair<std::string, int>> lines ;
  while(is>>k>>v) {
    lines.push_back(std::make_pair(k,v));
  }
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::shuffle(lines.begin(), lines.end(), std::default_random_engine(seed));
  for (int i = 0; i < size_; i++) {
    image_names_->push_back(lines[i].first);
    labels_.push_back(lines[i].second);
  }
  is.close();
  CHECK_GE(labels_.size(), size_) << "The size from conf file is " << size_
                                  << ", while the size from label file is "
                                  << labels_.size();
  return image_names_;
}
}  // namespace lapis
