// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-01 20:14

#include <glog/logging.h>

#include <boost/regex.hpp>
#include <boost/filesystem.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "disk/rgb_dir_source.h"

namespace lapis {
const std::string RGBDirSource::id_ = "RGBDirSource";

void RGBDirSource::Init(const DataSourceProto &ds_proto) {
  DataSource::Init(ds_proto);
  directory_ = ds_proto.path();
  width_ = ds_proto.width();
  height_ = ds_proto.height();
  record_length_ = 3 * width_ * height_;
  offset_ = ds_proto.offset();
  image_names_ = std::make_shared<StringVec>();
}

// the filename must end with valid image extension
bool isImage(const std::string &path) {
  static const boost::regex img_pattern(
    ".*\\.((jpg)|(jpeg))$",
    boost::regex::extended | boost::regex::icase);
  return boost::regex_match(path, img_pattern);
}

const std::shared_ptr<StringVec> &RGBDirSource::LoadData(
  const std::shared_ptr<StringVec> &keys) {
  LOG(INFO) << "In loadData func";
  if (keys != nullptr && !keys->empty()) {
    image_names_ = keys;
  } else {
    LOG(INFO) << "the dir is " << directory_;
    // assume all images are in a single plain folder
    boost::filesystem::directory_iterator iterator(directory_);
    for (; iterator != boost::filesystem::directory_iterator(); ++iterator) {
      // use filename instead of the whole path to reduce memory cost
      std::string filename = iterator->path().filename().string();
      if (isImage(filename))
        image_names_->push_back(filename);
    }
  }
  return image_names_;
}

void readImage(const std::string &path, int height, int width, float *val) {
  cv::Mat cv_img;
  if (height > 0 && width > 0) {
    cv::Mat cv_img_origin = cv::imread(path, CV_LOAD_IMAGE_COLOR);
    cv::resize(cv_img_origin, cv_img, cv::Size(height, width));
  } else {
    cv_img = cv::imread(path, CV_LOAD_IMAGE_COLOR);
  }
  CHECK(cv_img.data != NULL) << "Could not open or find file " << path;
  for (int c = 0; c < 3; ++c) {
    for (int h = 0; h < cv_img.rows; ++h) {
      for (int w = 0; w < cv_img.cols; ++w) {
        *val = static_cast<float>(cv_img.at<cv::Vec3b>(h, w)[c]);
        val++;
      }
    }
  }
}

void RGBDirSource::GetData(Blob *blob) {
  float *addr = blob->mutable_data();
  for (int i = 0; i < blob->num(); i++) {
    if (offset_ == size_)
      offset_ = 0;
    readImage(directory_ + "/" + image_names_->at(offset_), height_,
              width_,  &addr[i * record_length_]);
  }
}
}  // namespace lapis
