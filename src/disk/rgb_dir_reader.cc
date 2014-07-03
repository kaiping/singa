// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-01 20:14

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include "disk/rgb_dir_reader.h"

namespace lapis {
void RGBDirReader::init(const DataMetaProto& meta,
                        const vector<char*>& filenames = NULL) {
  path_ =  meta.path();
  height_ = meta.height();
  width_ = meta.width();
  if (filenames.empty()) {
    boost::filesystem::directory_iterator iterator(string(path_));
    for (; iterator != boots::filesystem::directory_iterator(); ++iterator)
      // since images are in the same folder,
      // use filename instead of the whole path to reduce memory cost
      filenames_.push_back(iterator->path().filename().c_str());
  } else {
    filenames_ = filenames;
  }
}

bool ends_with(std::string const& fullString, std::string const& ending) {
  if (fullString.length() >= ending.length()) {
    return (0 == fullString.compare(fullString.length() - ending.length(),
                                    ending.length(), ending));
  } else {
    return false;
  }
}

int RGBDirReader::next(string* k, string* v) {
  if (pos_ >= filenames_.size())
    return -1;
  // the filename must end with valid image extension
  if (!(ends_with(filenames_[pos_], ending_)))
    return 0;

  *k = filenames_[pos_];

  cv::Mat cv_img;
  if (height_ > 0 && width_ > 0) {
    cv::Mat cv_img_origin = cv::imread(path_+"/"+(*k), CV_LOAD_IMAGE_COLOR);
    cv::resize(cv_img_origin, cv_img, cv::Size(height, width));
  } else {
    cv_img = cv::imread(filename, CV_LOAD_IMAGE_COLOR);
  }
  if (!cv_img.data) {
    LOG(ERROR) << "Could not open or find file " << filename;
    return false;
  }
  RGBDatum datum;
  datum->set_channels(3);
  datum->set_height(cv_img.rows);
  datum->set_width(cv_img.cols);
  datum->clear_content();
  string* datum_string = datum->mutable_content();
  for (int c = 0; c < 3; ++c) {
    for (int h = 0; h < cv_img.rows; ++h) {
      for (int w = 0; w < cv_img.cols; ++w) {
        datum_string->push_back(
            static_cast<char>(cv_img.at<cv::Vec3b>(h, w)[c]));
      }
    }
  }
  datum.serializeToString(v);
  return v.length();
}
}  // namespace lapis
