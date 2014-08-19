// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-02 19:50

#include <glog/logging.h>
#include "datasource/label_source.h"
using std::shared_ptr;
namespace lapis {
const std::string LabelSource::type = "LabelSource";
const shared_ptr<StringVec> LabelSource::Init(const DataSourceProto &ds_proto,
      shared_ptr<StringVec>& filenames){
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
  for (int i = 0; i < size_; i++) {
    is >> k >> v;
    image_names_->push_back(k);
    labels_.push_back(static_cast<float>(v));
  }
  is.close();
  CHECK_GE(labels_.size(), size_) << "The size from conf file is " << size_
                                  << ", while the size from label file is "
                                  << labels_.size();
  return image_names_;
}
}  // namespace lapis
