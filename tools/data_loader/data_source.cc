#include <glog/logging.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <memory>
#include <fstream>

#include "utils/common.h"
#include "data_source.h"


const std::string ImageNetSource::type="ImageNetSource";
void ImageNetSource::Init(const string& folder, const string& meanfile, const int width, const int height){
  size_=0;
  offset_=0;
  width_=width;
  height_=height;
  record_size_=width*height*3;
  image_folder_=folder+"/img";
  label_path_=folder+"/rid.txt";
  LoadLabel(label_path_);
  mean_file_=meanfile;
  LoadMeanFile(mean_file_);
}
void ImageNetSource::LoadLabel(string path){
  LOG(INFO)<<"Loading labels...";
  std::ifstream is(path);
  CHECK(is.is_open()) << "Error open the label file " << label_path_;
  int v;
  std::string k;
  lines_.clear();
  if(size_>0){
    for (int i = 0; i < size_; i++) {
      is >> k >> v;
      lines_.push_back(std::make_pair(k,v));
    }
  }else{
    while(is >> k >> v)
      lines_.push_back(std::make_pair(k,v));
    size_=lines_.size();
  }
  is.close();
  LOG(INFO)<<"Load "<<lines_.size()<< " lines from label file";
}
void ImageNetSource::LoadMeanFile(string path){
  // read mean of the images
  singa::ReadProtoFromBinaryFile(path.c_str(), &data_mean_);
  LOG(INFO)<<"Read mean proto, of shape: "
    <<data_mean_.num()<<" "<<data_mean_.channels()
    <<" "<<data_mean_.height() <<" "<<data_mean_.width();
}

int ImageNetSource::ReadImage(const std::string &path, int height, int width,
    const float *mean, singa::DAryProto* image) {
  cv::Mat cv_img;
  if (height > 0 && width > 0) {
    cv::Mat cv_img_origin = cv::imread(path, CV_LOAD_IMAGE_COLOR);
    if(!cv_img_origin.data){
      LOG(ERROR)<<"invalid img "<<path;
      return 0;
    }
    cv::resize(cv_img_origin, cv_img, cv::Size(height, width));
  } else {
    cv_img = cv::imread(path, CV_LOAD_IMAGE_COLOR);
  }
  CHECK(cv_img.data != NULL) << "Could not open or find file " << path;
  image->add_shape(3);
  image->add_shape(cv_img.rows);
  image->add_shape(cv_img.cols);
  int idx=0;
  if(mean==nullptr){
    for (int c = 0; c < 3; ++c) {
      for (int h = 0; h < cv_img.rows; ++h) {
        for (int w = 0; w < cv_img.cols; ++w) {
          image->set_value(idx++,static_cast<float>(cv_img.at<cv::Vec3b>(h, w)[c]));
        }
      }
    }
  }else{
    for (int c = 0; c < 3; ++c) {
      for (int h = 0; h < cv_img.rows; ++h) {
        for (int w = 0; w < cv_img.cols; ++w) {
          image->set_value(idx++,static_cast<float>(cv_img.at<cv::Vec3b>(h, w)[c])-(*mean));
          mean++;
        }
      }
    }
  }
  return 1;
}

bool ImageNetSource::GetRecord(const int key, singa::Record* record) {
  if(key<0 || key>=size_)
    return false;
  singa::DAryProto *image=record->mutable_image();
  if(image->value().size()<record_size_){
    for(int i=0;i<record_size_;i++)
      image->add_value(0);
  }
  ReadImage(image_folder_ + "/" + lines_.at(key).first, height_,
            width_, data_mean_.data().data(),image);
  record->set_label(lines_.at(key).second);
  return true;
}

bool ImageNetSource::NextRecord(string* key, singa::Record *record) {
  singa::DAryProto *image=record->mutable_image();
  if(image->value().size()<record_size_){
    for(int i=image->value().size();i<record_size_;i++)
      image->add_value(0);
  }
  *key=lines_.at(offset_).first;
  int ret=ReadImage(image_folder_ + "/" + *key, height_, width_, data_mean_.data().data(),image);
  record->set_label(lines_.at(offset_).second);
  offset_++;
  return ret;
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
