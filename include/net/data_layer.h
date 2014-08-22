// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-14 22:37

#ifndef INCLUDE_MODEL_DATA_LAYER_H_
#define INCLUDE_MODEL_DATA_LAYER_H_
#include <string>
#include <vector>
#include <map>
#include "net/layer.h"
#include "net/trainer.h"
#include "net/lapis.h"
#include "proto/model.pb.h"


namespace lapis {
/**
 * Layer for fetching raw input features
 * It setups DataSource firstlyn ::Setup() and then fetch the data batch in
 * ::Forward().
 */
class DataLayer : public Layer {
 public:
  /**
   * Identifier of this layer, the value is "Data".
   */
  static const std::string kType;
  /**
   * Set data source identifier, i.e. name.
   */
  virtual void Init(const LayerProto &proto);
  /**
   * Set the input batch shape, including batchsize, channels, height, width.
   * @param shape
   */
  void SetInputShape(int batchsize, const Shape &data_shape);
  void SetInputStore(int store_id);
  void SetData(const Blob &blob);
  /**
   * allocate memory
   */
  virtual void Setup(const char flag);
  /**
   * fetch data from data source
   */
  virtual void Forward();
  /*
   * Just call Backward function of out going edges.
   */
  virtual void Backward();
  /**
   * Write the data source name
   */
  virtual void ToProto(LayerProto *layer_proto);
  virtual bool HasInput() {
    return true;
  }
  /**
   * @param edge if edge is nullptr it means this function is called to fill
   * new records for the layer. hence return the tmp blob if the image should
   * be croped; otherwise return the data blob
   */
  virtual Blob &feature(Edge *edge) {
    if(edge==nullptr&& cropsize_)
      return tmp_;
    else
      return data_;
  }
  /**
   * Because DataLayer is usually connected from loss edges, hence this
   * function returns the data provided by DataSource to the loss edge to
   * compute the gradients.
   * @param edge not used currently.
   */
  virtual Blob &gradient(Edge *edge) {
    return data_;
  }

  inline int store_id() {
    return  store_id_;
  }

  inline std::string& data_source() {
    return data_source_;
  }

 private:
  bool mirror_;
  int cropsize_;
  int batchsize_, channels_,height_,width_;
  Blob data_, tmp_;
  std::string data_source_;
  int store_id_;
};

}  // namespace lapis
#endif  // INCLUDE_MODEL_DATA_LAYER_H_
