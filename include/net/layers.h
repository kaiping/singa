// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-09-10 16:14

#ifndef INCLUDE_MODEL_LAYERS_H_
#define INCLUDE_MODEL_LAYERS_H_
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
  void LoadData(const DAry &input, Phase phase);
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
  virtual const Blob &data(Edge *edge) {
      return data_;
  }
  /**
   * Because DataLayer is usually connected from loss edges, hence this
   * function returns the data provided by DataSource to the loss edge to
   * compute the gradients.
   * @param edge not used currently.
   */
  virtual const Blob &grad(Edge *edge) {
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
  std::string data_source_;
  int store_id_;
};

}  // namespace lapis
#endif  // INCLUDE_MODEL_LAYERS_H_

