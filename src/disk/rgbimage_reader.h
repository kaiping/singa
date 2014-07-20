// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-01 20:14

#include <boost/regex.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include "disk/rgbimage_reader.h"

namespace lapis {
void RGBImageReader::Init(const DataSourceProto &ds_proto,
                          const std::vector<std::string> &path_suffix,
                          int offset =0) {
  path_prefix_=ds_proto.path();
  width_=ds_proto.width();
  height=ds_proto.height();
  if (path_suffix.empty()) {
    // assume all images are in a single plain folder
    boost::filesystem::directory_iterator iterator(path_prefix);
    for (; iterator != boost::filesystem::directory_iterator(); ++iterator)
      // use filename instead of the whole path to reduce memory cost
      string filename=iterator->path().filename();
      if (isImage(filename))
        path_suffix_.push_back(filename);
  } else {
    path_suffix_=path_suffix;
  }
  it_=path_suffix_.begin()+offset;
}

// the filename must end with valid image extension
bool isImage(const std::string &path) {
  static const boost::regex img_pattern(
      ".*\.((jpg)|(jpeg))$",
      boost::regex::extended|boost::regex::icase);
  return boost::regex_match(path, img_pattern);
}

bool RGBImageReader::ReadNextRecord(std::string *key, float *val) {
  if (it_==path_suffix_.end())
    return false;
  *key = *it;
  cv::Mat cv_img;
  string full_path=path_prefix_ + "/" + (*it);
  if (height_ > 0 && width_ > 0) {
    cv::Mat cv_img_origin = cv::imread(full_path, CV_LOAD_IMAGE_COLOR);
    cv::resize(cv_img_origin, cv_img, cv::Size(height, width));
  } else {
    cv_img = cv::imread(full_path, CV_LOAD_IMAGE_COLOR);
  }
  CHECK_NOTNULL(cv_img.data) << "Could not open or find file "
                                << full_path <<"\n";
  for (int c = 0; c < 3; ++c) {
    for (int h = 0; h < cv_img.rows; ++h) {
      for (int w = 0; w < cv_img.cols; ++w) {
        *val=static_cast<float>(cv_img.at<cv::Vec3b>(h, w)[c]);
        val++;
      }
    }
  }
  return true;
}
void RGBImageReader::Reset() {
  it_=path_suffix_.begin();
}
int RGBImageReader::Offset() {
  return it_-path_suffix_.begin();
}

}  // namespace lapis
