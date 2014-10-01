// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-16 22:01
#include <glog/logging.h>
#include <memory>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "datasource/data_source.h"

namespace lapis {
/*****************************************************************************
 * Implementation for DataSource
 ****************************************************************************/
std::map<string, Shape> DataSource::ShapesOf(const DataSourceProtos &sources) {
  std::map<string, Shape> shape_map;
  for(auto& source: sources) {
    shape_map[source.name()]=source.shape();
  }
  return shape_map;
}

const DataSource::Init(const DataSourceProto &ds_proto){
  if(proto.shape().has_num())
    size_ = proto.shape().num();
  else
    size_=0;
  name_ = proto.name();
  offset_ = proto.offset();
}

void DataSource::ToProto(DataSourceProto *proto) {
  proto->set_offset(offset_);
}

const void ImageNetSource::Init(const DataSourceProto &proto){
  DataSource::Init(proto);
  image_folder_ = proto.image_folder();
  label_path_=proto.label_path();
  if(ds_proto.has_mean_file())
    mean_file_=ds_proto.mean_file();
  else
    mean_file_="";
  Shape s=proto.shape();
  width_ = s.width();
  height_ = s.height();
  channels_=s.channels();
  record_size_=width_*height_*channels_;
  CHECK_EQ(channels_,3);
  LOG(INFO)<<"Loading labels...";
  std::ifstream is(label_path_);
  CHECK(is.is_open()) << "Error open the label file " << label_path_;
  int v;
  std::string k;
  image_names_ = std::make_shared<StringVec>();
  if(size_>0){
    for (int i = 0; i < size_; i++) {
      is >> k >> v;
      lines_.push_back(std::make_pair(k,v));
    }
  }else{
    while(is >> k >> v)
      lines_.push_back(std::make_pair(k,v));
  }
  is.close();
  LOG(INFO)<<"Load "<<labels_.size();
  if(do_shuffle_){
    DLOG(INFO)<<"Do Shuffling...";
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::shuffle(lines_.begin(), lines_.end(), std::default_random_engine(seed));
  }
  // read mean of the images
  if(mean_file_.length()) {
    data_mean_=new MeanProto();
    ReadProtoFromBinaryFile(mean_file_.c_str(), data_mean_);
    VLOG(1)<<"read mean proto, of shape: "
      <<data_mean_->num()<<" "<<data_mean_->channels()
      <<" "<<data_mean_->height() <<" "<<data_mean_->width();
  }
}

void ImageNetSource::ReadImage(const std::string &path, int height, int width,
    const float *mean, Datum* datum) {
  cv::Mat cv_img;
  if (height > 0 && width > 0) {
    VLOG(3)<<"resize image";
    cv::Mat cv_img_origin = cv::imread(path, CV_LOAD_IMAGE_COLOR);
    cv::resize(cv_img_origin, cv_img, cv::Size(height, width));
  } else {
    VLOG(3)<<"no image resize";
    cv_img = cv::imread(path, CV_LOAD_IMAGE_COLOR);
  }
  CHECK(cv_img.data != NULL) << "Could not open or find file " << path;
  datum->set_channels(3);
  datum->set_height(cv_img.rows);
  datum->set_width(cv_img.cols);
  float* dptr=datum->mutable_value();
  if(mean==nullptr){
    for (int c = 0; c < 3; ++c) {
      for (int h = 0; h < cv_img.rows; ++h) {
        for (int w = 0; w < cv_img.cols; ++w) {
          *dptr=static_cast<float>(cv_img.at<cv::Vec3b>(h, w)[c]);
          dptr++;
        }
      }
    }
  }else{
    for (int c = 0; c < 3; ++c) {
      for (int h = 0; h < cv_img.rows; ++h) {
        for (int w = 0; w < cv_img.cols; ++w) {
          *dptr=static_cast<float>(cv_img.at<cv::Vec3b>(h, w)[c])-(*mean);
          dptr++;
          mean++;
        }
      }
    }
  }
}

void RGBDirSource::NextRecord(ImageNetRecord *record) {
  Datum *datum=record->mutable_image();
  if(datum->value().size()<record_size_){
    for(int i=0;i<ds->channels()*ds->with()*ds->height();i++)
      datum->add_value(0);
  }
  ReadImage(img_folder_ + "/" + lines_.at(offset_).first, height_,
            width_, data_mean_->data().data(),datum);
  record->set_label(lines_.at(offset_).second;
  offset_++;
}
/*****************************************************************************
 * Implementation of DataSourceFactory
 ****************************************************************************/
#define CreateDS(DSClass) [](void)->DataSource* {return new DSClass();}

std::shared_ptr<DataSourceFactory> DataSourceFactory::instance_;

std::shared_ptr<DataSourceFactory> DataSourceFactory::Instance() {
   if (!instance_.get())
     instance_.reset(new DataSourceFactory());
   return instance_;
}

DataSourceFactory::DataSourceFactory() {
  RegisterCreateFunction(ImageNetSource::type, CreateDS(ImageNetSource));
}

void DataSourceFactory::RegisterCreateFunction(
  const string &id,
  std::function<DataSource*(void)> create_function) {
  ds_map_[id] = create_function;
  DLOG(INFO)<<"register DataSource: "<<id;
}

DataSource *DataSourceFactory::Create(const string id) {
  CHECK(ds_map_.find(id) != ds_map_.end()) << "The reader " << id
      << " has not been registered\n";
  return ds_map_.at(id)();
}
}  // namespace lapis
