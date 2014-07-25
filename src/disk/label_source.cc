// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-02 19:50

#include <glog/logging.h>
#include "disk/label_source.h"
namespace lapis {
const std::string LabelSource::id_ = "LabelSource";

void LabelSource::Init(const DataSourceProto &ds_proto) {
  DataSource::Init(ds_proto);
  label_path_ = ds_proto.path();
  image_names_ = std::make_shared<StringVec>();
}

void LabelSource::ToProto(DataSourceProto *ds_proto) {
  ds_proto->set_path(label_path_);
}

void LabelSource::GetData(Blob *blob) {
  float *addr = blob->mutable_data();
  for (int i = 0; i < blob->num(); i++) {
    if (offset_ == size_)
      offset_ = 0;
    addr[i] = labels_[offset_++];
  }
}

const std::shared_ptr<StringVec> &LabelSource::LoadData(
  const std::shared_ptr<StringVec> &keys) {
  std::ifstream is(label_path_);
  CHECK(is.is_open()) << "Error open the label file " << label_path_;
  std::string k;
  int v;
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
