// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-14 22:37

#ifndef INCLUDE_MODEL_DATA_LAYER_H_
#define INCLUDE_MODEL_DATA_LAYER_H_
#include <string>
#include <vector>
#include <map>
#include "model/layer.h"
#include "model/trainer.h"


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
   * Setup the data source/provider.
   * ::Forward() function will fetch one batch of data from this data source
   * @param batchsize num of instance in one batch
   * @param alg
   * @param sources, a vector of DataSource objects (e.g., rgb feature or
   * label) from which this layer will select one based on data source
   * identifier, i.e., name.
   */
  virtual void Setup(int batchsize, TrainerProto::Algorithm alg,
                     const std::vector<DataSource *> &sources);
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
  virtual bool HasInput() { return true; }
  /**
   * Return the data provided by DataSource
   * @param edge not used currently.
   */
  virtual Blob4 *feature(Edge *edge) { return &data_; }
  /**
   * Because DataLayer is usually connected from loss edges, hence this
   * function returns the data provided by DataSource to the loss edge to
   * compute the gradients.
   * @param edge not used currently.
   */
  virtual Blob4 *gradient(Edge *edge) { return &data_; }

  /** identifier for this layer.
   * LayerFactory will create an instance of this based on this identifier
   */

 private:
  Blob4 data_;
  DataSource *data_source_;
  std::string data_source_name_;
};

}  // namespace lapis
#endif  // INCLUDE_MODEL_DATA_LAYER_H_
