#include <gtest/gtest.h>
#include <model/neuralnet.h>
#include "proto/model.pb.h"

using namespace singa;

NetProto CreateNetProto(){
  NetProto proto;
  LayerProto *layer;

  layer=proto.add_layer();
  layer->set_name("data");
  layer->set_type("kShardData");
  DataProto *data=layer->mutable_data_param();
  data->set_batchsize(8);

  // 4x3x10x10
  layer=proto.add_layer();
  layer->set_name("rgbimage");
  layer->set_type("kRGBImage");
  layer->add_srclayers("data");

  // 4x1
  layer=proto.add_layer();
  layer->set_name("label");
  layer->set_type("kLabel");
  layer->add_srclayers("data");

  // 4x8x9x9
  layer=proto.add_layer();
  layer->set_name("conv1");
  layer->set_type("kConvolution");
  layer->add_srclayers("rgbimage");
  ConvolutionProto *conv=layer->mutable_convolution_param();
  conv->set_num_output(8);
  conv->set_kernel_h(2);
  conv->set_kernel_w(2);

  // 4x8x9x9
  layer=proto.add_layer();
  layer->set_name("relu1");
  layer->set_type("kReLU");
  layer->add_srclayers("conv1");

  // 4x8x4x4
  layer=proto.add_layer();
  layer->set_name("pool1");
  layer->set_type("kPooling");
  layer->add_srclayers("relu1");
  PoolingProto *pool=layer->mutable_pooling_param();
  pool->set_kernel_size(4);
  pool->set_stride(2);

  // 4x10
  layer=proto.add_layer();
  layer->set_name("fc1");
  layer->set_type("kInnerProduct");
  layer->add_srclayers("pool1");
  InnerProductProto *inner=layer->mutable_inner_product_param();
  inner->set_num_output(10);

  // 4x10
  layer=proto.add_layer();
  layer->set_name("loss");
  layer->set_type("kSoftmaxLoss");
  layer->add_srclayers("fc1");
  layer->add_srclayers("label");

  return proto;
}

TEST(NeuralNetTest, NoPartition){
  NetProto proto=CreateNetProto();
  NeuralNet net(proto);
  const auto& layers=net.layers();
  ASSERT_EQ(8, layers.size());
  ASSERT_EQ("data", layers.at(0)->name());
  ASSERT_EQ("loss", layers.at(7)->name());
}

TEST(NeuralNetTest, DataPartition){
  NetProto proto=CreateNetProto();
  proto.set_partition_type(kDataPartition);
  NeuralNet net(proto, 3);
  const auto& layers=net.layers();
  ASSERT_EQ(28, layers.size());
  ASSERT_EQ("data", layers.at(0)->name());
}
TEST(NeuralNetTest, LayerPartition){
  NetProto proto=CreateNetProto();
  proto.set_partition_type(kLayerPartition);
  NeuralNet net(proto, 2);
 // const auto& layers=net.layers();
}
TEST(NeuralNetTest, HyridPartition){


}
