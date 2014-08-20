// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-01 20:14

#include <glog/logging.h>
#include <boost/regex.hpp>
#include <boost/filesystem.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "datasource/rgb_dir_source.h"
#include "utils/proto_helper.h"

namespace lapis {
const std::string RGBDirSource::type = "RGBDirSource";

const std::shared_ptr<StringVec> RGBDirSource::Init(const DataSourceProto &ds_proto,
      std::shared_ptr<StringVec>& filenames){
  VLOG(3)<<"init rgb source";
  DataSource::Init(ds_proto, filenames);
  directory_ = ds_proto.path();
  if(ds_proto.has_mean_file())
    mean_file_=ds_proto.mean_file();
  else
    mean_file_="";
  Shape s=ds_proto.shape();
  width_ = s.width();
  height_ = s.height();
  channels_=s.channels();
  CHECK_EQ(channels_,3);
  image_size_ = 3 * width_ * height_;
  return LoadData(filenames);
}

// the filename must end with valid image extension
bool isImage(const std::string &path) {
  static const boost::regex img_pattern(
    ".*\\.((jpg)|(jpeg))$",
    boost::regex::extended | boost::regex::icase);
  return boost::regex_match(path, img_pattern);
}

const std::shared_ptr<StringVec> RGBDirSource::LoadData(
  const std::shared_ptr<StringVec> &keys) {
  DLOG(INFO) << "Load RGB Data (collect img names) ";
  if (keys){
    if(!keys->empty())
      image_names_ = keys;
    VLOG(3)<<"copy file names";
  } else {
    image_names_ = std::make_shared<StringVec>();
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
  // read mean of the images
  if(mean_file_.length()) {
    VLOG(3)<<"mean file path "<<mean_file_;
    ReadProtoFromBinaryFile(mean_file_.c_str(), &data_mean_);
    VLOG(2)<<"read mean proto, of shape: "
      <<data_mean_.num()<<" "<<data_mean_.channels()
      <<" "<<data_mean_.height() <<" "<<data_mean_.width();
  }
  return image_names_;
}

void readImage(const std::string &path, int height, int width,
               const float *mean, float *val) {
  cv::Mat cv_img;
  if (height > 0 && width > 0) {
    cv::Mat cv_img_origin = cv::imread(path, CV_LOAD_IMAGE_COLOR);
    cv::resize(cv_img_origin, cv_img, cv::Size(height, width));
  } else {
    cv_img = cv::imread(path, CV_LOAD_IMAGE_COLOR);
  }
  CHECK(cv_img.data != NULL) << "Could not open or find file " << path;
  if(mean==nullptr){
    for (int c = 0; c < 3; ++c) {
      for (int h = 0; h < cv_img.rows; ++h) {
        for (int w = 0; w < cv_img.cols; ++w) {
          *val = static_cast<float>(cv_img.at<cv::Vec3b>(h, w)[c]);
          val++;
        }
      }
    }
  }else{
    for (int c = 0; c < 3; ++c) {
      for (int h = 0; h < cv_img.rows; ++h) {
        for (int w = 0; w < cv_img.cols; ++w) {
          *val = static_cast<float>(cv_img.at<cv::Vec3b>(h, w)[c])-(*mean);
          val++;
          mean++;
        }
      }
    }
  }
}

void RGBDirSource::NextRecord(FloatVector *record) {
  VLOG(3)<<"GetData";
  readImage(directory_ + "/" + image_names_->at(offset_), height_,
              width_, data_mean_.data().data(),
              record->mutable_data()->mutable_data());
  offset_++;
}

void RGBDirSource::GetData(Blob *blob) {
  VLOG(3)<<"GetData";
  CHECK_EQ(blob->height(), height_);
  CHECK_EQ(blob->width(), width_);
  CHECK_EQ(blob->channels(), channels_);
  VLOG(3)<<"After check";
  float *addr = blob->dptr;
  for (int i = 0; i < blob->num(); i++) {
    if (offset_ == size_)
      offset_ = 0;
    readImage(directory_ + "/" + image_names_->at(offset_), height_,
              width_, data_mean_.data().data(), &addr[i * image_size_]);
  }
}
}  // namespace lapis
