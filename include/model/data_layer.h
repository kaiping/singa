// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-14 22:37

#ifndef INCLUDE_MODEL_DATA_LAYER_H_
#define INCLUDE_MODEL_DATA_LAYER_H_
#include <string>
#include <vector>
#include <map>
#include "model/layer.h"

namespace lapis {
/**
 * Layer for fetching raw input features
 * It setups DataSource firstly and the fetch the data batch in ::Forward()
 * Since no parameters for this layer, ::ComputeParamUpdates() is empty
 */
class DataLayer : public Layer {
 public:
  static const string kDataLayer = "Data";
  virtual void Init(const LayerProto &layer_proto,
                    const map<string, Edge *> &edge_map);
  /**
   * setup the data source/provider.
   * ::Forward() function will fetch one batch of data from this data source
   * @param ds pointer to a DataSource object, which can provide raw feature or
   * labels.
   */
  virtual void Setup(int batchsize, Trainer::Algorithm alg,
                     const vector<DataSource *> &sources);
  /**
   * fetch data from data source
   */
  virtual void Forward();
  virtual void Backward();
  virtual void ComputeParamUpdates(const Trainer *trainer);
  virtual void ToProto(LayerProto *layer_proto);
  virtual inline bool HasInput();
  /**
   * Return the data provided by DataSource
   * @param edge not used currently.
   */
  virtual inline Blob &Feature(Edge *edge);
  /**
   * Because DataLayer is usually connected from loss edges, hence this
   * function returns the data provided by DataSource to the loss edge to
   * compute the gradients.
   * @param edge not used currently.
   */
  virtual inline Blob &Gradient(Edge *edge);

  /** identifier for this layer.
   * LayerFactory will create an instance of this based on this identifier
   */

 private:
  Blob data_;
  DataSource *data_source_;
}

}  // namespace lapis
#endif  // INCLUDE_MODEL_DATA_LAYER_H_

