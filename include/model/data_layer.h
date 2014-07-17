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
 * It setups DataSource firstly and the fetch the data batch in ::Forward()
 * Since no parameters for this layer, ::ComputeParamUpdates() is empty
 */
class DataLayer : public Layer {
 public:
  /**
   * Identifier of this layer, the value is "Data".
   */
  static const std::string kDataLayer;
  virtual void Init(const LayerProto &layer_proto,
                    const std::map<std::string, Edge *> &edge_map);
  /**
   * setup the data source/provider.
   * ::Forward() function will fetch one batch of data from this data source
   * @param ds pointer to a DataSource object, which can provide raw feature or
   * labels.
   */
  virtual void Setup(int batchsize, TrainAlgorithm alg,
                     const std::vector<DataSource *> &sources);
  /**
   * fetch data from data source
   */
  virtual void Forward();
  virtual void Backward();
  virtual void ComputeParamUpdates(const Trainer *trainer);
  virtual void ToProto(LayerProto *layer_proto);
  virtual bool HasInput() {
    return true;
  }
  /**
   * Return the data provided by DataSource
   * @param edge not used currently.
   */
  virtual Blob *Feature(Edge *edge) {
    return &data_;
  }
  /**
   * Because DataLayer is usually connected from loss edges, hence this
   * function returns the data provided by DataSource to the loss edge to
   * compute the gradients.
   * @param edge not used currently.
   */
  virtual Blob *Gradient(Edge *edge) {
    return &data_;
  }

  /** identifier for this layer.
   * LayerFactory will create an instance of this based on this identifier
   */

 private:
  Blob data_;
  DataSource *data_source_;
  std::string data_source_name_;
};

}  // namespace lapis
#endif  // INCLUDE_MODEL_DATA_LAYER_H_

