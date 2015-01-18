#include <glog/logging.h>
#include <fcntl.h>
#include <google/protobuf/message.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/coded_stream.h>
#include <memory>
#include <fstream>
using google::protobuf::Message;
using google::protobuf::io::FileInputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::CodedInputStream;
#include "data_source.h"

/************************************************************************
 * Implement DataSouce for ImageNet input
 ************************************************************************/
uint32_t swap_endian(uint32_t val) {
  val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
  return (val << 16) | (val >> 16);
}

void MnistSource::Init(string imagefile, string labelfile){
  //Open files
  imagestream_.open(imagefile, std::ios::in | std::ios::binary);
  labelstream_.open(labelfile, std::ios::in | std::ios::binary);
  CHECK(imagestream_.is_open()) << "Unable to open file " << imagefile;
  CHECK(labelstream_.is_open()) << "Unable to open file " << labelfile;
  // Read the magic and the meta data
  uint32_t magic;
  uint32_t num_items;
  uint32_t num_labels;
  imagestream_.read(reinterpret_cast<char*>(&magic), 4);
  magic = swap_endian(magic);
  CHECK_EQ(magic, 2051) << "Incorrect image file magic.";
  labelstream_.read(reinterpret_cast<char*>(&magic), 4);
  magic = swap_endian(magic);
  CHECK_EQ(magic, 2049) << "Incorrect label file magic.";
  imagestream_.read(reinterpret_cast<char*>(&num_items), 4);
  num_items = swap_endian(num_items);
  labelstream_.read(reinterpret_cast<char*>(&num_labels), 4);
  num_labels = swap_endian(num_labels);
  CHECK_EQ(num_items, num_labels);
  size_=num_items;
  imagestream_.read(reinterpret_cast<char*>(&height_), 4);
  height_ = swap_endian(height_);
  imagestream_.read(reinterpret_cast<char*>(&width_), 4);
  width_ = swap_endian(width_);
  image_=new char[height_*width_];
  LOG(ERROR)<<"Data info, num of instances: "<<size_<<" height:"<<height_
    <<" width_:"<<width_;
}

bool MnistSource::NextRecord(string* key, singa::Record *record){
  if(!eof()){
    offset_++;
    char label;
    imagestream_.read(image_, height_*width_);
    labelstream_.read(&label, 1);
    //use imagenetrecord here
    record->set_type(singa::Record_Type_kMnist);
    singa::MnistRecord* rec=record->mutable_mnist();
    rec->set_label(static_cast<int>(label));
    string pixel;
    pixel.resize(height_*width_);
    for(int i=0;i<height_*width_;i++)
      pixel[i]=image_[i];
    rec->set_pixel(pixel);
    return true;
  }else{
    return false;
  }
}

/************************************************************************
 * Implement DataSouce for ImageNet input
 ************************************************************************/
void ReadProtoFromBinaryFile(const char* filename, Message* proto) {
  VLOG(3)<<"read from binry file";
  int fd = open(filename, O_RDONLY);
  CHECK_NE(fd, -1) << "File not found: " << filename;
  ZeroCopyInputStream* raw_input = new FileInputStream(fd);
  CodedInputStream* coded_input = new CodedInputStream(raw_input);
  coded_input->SetTotalBytesLimit(536870912, 268435456);
  VLOG(3)<<"before parse";
  CHECK(proto->ParseFromCodedStream(coded_input));
  delete coded_input;
  delete raw_input;
  close(fd);
  VLOG(3)<<"read binry file";
}

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
  ReadProtoFromBinaryFile(path.c_str(), &data_mean_);
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
  singa::ImageNetRecord *imagenet=record->mutable_imagenet();
  singa::DAryProto *image=imagenet->mutable_image();
  if(image->value().size()<record_size_){
    for(int i=0;i<record_size_;i++)
      image->add_value(0);
  }
  ReadImage(image_folder_ + "/" + lines_.at(key).first, height_,
            width_, data_mean_.data().data(),image);
  imagenet->set_label(lines_.at(key).second);
  return true;
}

bool ImageNetSource::NextRecord(string* key, singa::Record *record) {
  record->set_type(singa::Record_Type_kImageNet);
  singa::ImageNetRecord *imagenet=record->mutable_imagenet();
  singa::DAryProto *image=imagenet->mutable_image();
  if(image->value().size()<record_size_){
    for(int i=image->value().size();i<record_size_;i++)
      image->add_value(0);
  }
  *key=lines_.at(offset_).first;
  int ret=ReadImage(image_folder_ + "/" + *key, height_, width_, data_mean_.data().data(),image);
  imagenet->set_label(lines_.at(offset_).second);
  offset_++;
  return ret;
}
