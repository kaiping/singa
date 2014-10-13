// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-16 22:01
#include <hdfs.h>
#include <glog/logging.h>
#include <boost/filesystem.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <memory>
#include <fstream>


#include "datasource/data_source.h"
#include "utils/proto_helper.h"


namespace lapis {
/*****************************************************************************
 * Implementation for DataSource
 ****************************************************************************/
void DataSource::Init(const DataSourceProto &proto){
  if(proto.has_size())
    size_=proto.size();
  else
    size_=0;
  name_ = proto.name();
  offset_ = proto.offset();
}

void DataSource::ToProto(DataSourceProto *proto) {
  proto->set_offset(offset_);
}

const std::string ImageNetSource::type="ImageNetSource";
void ImageNetSource::Init(const DataSourceProto &proto, const ShardProto& shard){
  DataSource::Init(proto);
  DataSourceProto::Shape shape=proto.shape(0);
  width_ = shape.s(2);
  height_ = shape.s(1);
  channels_=shape.s(0);
  record_size_=width_*height_*channels_;
  CHECK_EQ(channels_,3);

  hdfs_=proto.hdfs();
  label_path_=proto.label_path();
  mean_file_=proto.mean_file();
  string local_label_path=shard.shard_folder()+"/"+name_+"_label";
  string local_mean_file=shard.shard_folder()+"/"+name_+"_mean";
  if(hdfs_){
    CopyFileFromHDFS(label_path_, local_label_path);
    CopyFileFromHDFS(mean_file_, local_mean_file);
  }
  label_path_=local_label_path;
  mean_file_=local_mean_file;
  LoadLabel(label_path_);
  LoadMeanFile(mean_file_);

  CHECK_GE(size_, shard.record_size());
  size_=shard.record_size();
  vector<std::pair<string, int>> tmp=lines_;
  vector<std::string> imgnames;
  lines_.clear();
  for(auto rid: shard.record()){
    lines_.push_back(tmp[rid]);
    imgnames.push_back(tmp[rid].first);
  }
  image_folder_=proto.image_folder();
  string local_image_folder=shard.shard_folder()+"/"+name_+"_img";
  if(hdfs_){
    CopyFilesFromHDFS(image_folder_, local_image_folder , imgnames);
  }
  image_folder_=local_image_folder;
}
void ImageNetSource::Init(const DataSourceProto &proto){
  DataSource::Init(proto);
  DataSourceProto::Shape shape=proto.shape(0);
  width_ = shape.s(2);
  height_ = shape.s(1);
  channels_=shape.s(0);
  record_size_=width_*height_*channels_;
  CHECK_EQ(channels_,3);

  //do_shuffle_=proto.shuffle();
  image_folder_ = proto.image_folder();
  label_path_=proto.label_path();
  mean_file_=proto.mean_file();
  LoadLabel(label_path_);
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
  LOG(INFO)<<"Load "<<lines_.size();
}
void ImageNetSource::LoadMeanFile(string path){
  // read mean of the images
  data_mean_=new MeanProto();
  ReadProtoFromBinaryFile(path.c_str(), data_mean_);
  LOG(INFO)<<"Read mean proto, of shape: "
    <<data_mean_->num()<<" "<<data_mean_->channels()
    <<" "<<data_mean_->height() <<" "<<data_mean_->width();
}
void ImageNetSource::Reset(const ShardProto& shard) {
  CHECK_GE(size_, shard.record_size());
  vector<std::pair<string, int>> tmp=lines_;
  lines_.clear();
  for(auto rid: shard.record()){
    lines_.push_back(tmp[rid]);
  }
  /*
  if(hdfs_){
    // copy files from hdfs to local shard folder
    // will read images from shard folder later to create level db
    CopyFilesFromHDFS(sp.shard_folder());
    image_folder_=sp.shard_folder();
    hdfs_=false;
  }else{
    // else read images from the image_folder
    // which must be shared on all workers
    boost::filesystem::path p(image_folder_);
    CHECK(boost::filesystem::exists(p));
  }
  */
}
int ImageNetSource::CopyFileFromHDFS(string hdfs_path, string local_path) {
  LOG(INFO)<<"Copy file from hdfs: "<<hdfs_path<<" to local: "<<local_path;
  hdfsFS fs=hdfsConnect("default", 0);
  if(!fs) {
    LOG(ERROR)<<"Oops! Failed to connect to hdfs!";
  }
  hdfsFS lfs = hdfsConnect(NULL, 0);
  if(!lfs) {
    fprintf(stderr, "Oops! Failed to connect to 'local' hdfs!\n");
    exit(-1);
  }
  int ret=(!hdfsCopy(fs, hdfs_path.c_str(), lfs, local_path.c_str()));
  CHECK(ret);
  hdfsDisconnect(fs);
  hdfsDisconnect(lfs);
  return ret;
}

int ImageNetSource::CopyFilesFromHDFS(string hdfs_folder, string local_folder,
    std::vector<string> files) {
  LOG(INFO)<<"Copy files from hdfs: "<<hdfs_folder<<" to local: "<<local_folder;
  hdfsFS fs=hdfsConnect("default", 0);
  if(!fs) {
    LOG(ERROR)<<"Oops! Failed to connect to hdfs!";
  }
  hdfsFS lfs = hdfsConnect(NULL, 0);
  if(!lfs) {
    fprintf(stderr, "Oops! Failed to connect to 'local' hdfs!\n");
    exit(-1);
  }
  boost::filesystem::path dir_path(local_folder);
  if(boost::filesystem::create_directories(dir_path)) {
    LOG(INFO)<<"create shard folder "<<local_folder;
  }
  int ncopy=0;
  for(auto& file: files){
    string hdfs_path=hdfs_folder+"/"+file;
    string local_path=local_folder+"/"+file;
    if(!hdfsCopy(fs, hdfs_path.c_str(), lfs, local_path.c_str()))
      ncopy++;
    else
      LOG(INFO)<<"Failed to Copy "<<hdfs_path<<" to "<<local_path;
    if(ncopy%100==0)
      LOG(INFO)<<"Have copied "<<ncopy<<" files";
  }
  hdfsDisconnect(fs);
  hdfsDisconnect(lfs);
  return ncopy;
}
void ImageNetSource::ReadImage(const std::string &path, int height, int width,
    const float *mean, DAryProto* image) {
  cv::Mat cv_img;
  if (height > 0 && width > 0) {
    cv::Mat cv_img_origin = cv::imread(path, CV_LOAD_IMAGE_COLOR);
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
}

bool ImageNetSource::GetRecord(const int key, Record* record) {
  if(key<0 || key>=size_)
    return false;
  DAryProto *image=record->mutable_image();
  if(image->value().size()<record_size_){
    for(int i=0;i<record_size_;i++)
      image->add_value(0);
  }
  ReadImage(image_folder_ + "/" + lines_.at(key).first, height_,
            width_, data_mean_->data().data(),image);
  record->set_label(lines_.at(key).second);
  return true;
}

void ImageNetSource::NextRecord(string* key, Record *record) {
  DAryProto *image=record->mutable_image();
  if(image->value().size()<record_size_){
    for(int i=0;i<record_size_;i++)
      image->add_value(0);
  }
  *key=lines_.at(offset_).first;
  ReadImage(image_folder_ + "/" + *key, height_,
            width_, data_mean_->data().data(),image);
  record->set_label(lines_.at(offset_).second);
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
