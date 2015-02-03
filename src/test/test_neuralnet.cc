#include <gtest/gtest.h>
#include <model/neuralnet.h>
#include "proto/model.pb.h"

using namespace singa;

NetProto CreateNetProto(){
  NetProto proto;
  LayerProto *layer;

  // 4x3x10x10
  layer=proto.add_layer();
  layer->set_name("rgbimage");
  layer->set_type("RGBImage");
  RGBProto *rgb=layer->mutable_rgb_param();
  rgb->add_shape(4);
  rgb->add_shape(3);
  rgb->add_shape(10);
  rgb->add_shape(10);

  // 4x1
  layer=proto.add_layer();
  layer->set_name("label");
  layer->set_type("Label");

  // 4x8x9x9
  layer=proto.add_layer();
  layer->set_name("conv1");
  layer->set_type("Convolution");
  layer->add_srclayers("rgbimage");
  ConvolutionProto *conv=layer->mutable_convolution_param();
  conv->set_num_output(8);
  conv->set_kernel_h(2);
  conv->set_kernel_w(2);

  // 4x8x9x9
  layer=proto.add_layer();
  layer->set_name("relu1");
  layer->set_type("ReLU");
  layer->add_srclayers("conv1");

  // 4x8x4x4
  layer=proto.add_layer();
  layer->set_name("pool1");
  layer->set_type("Pooling");
  layer->add_srclayers("relu1");
  PoolingProto *pool=layer->mutable_pooling_param();
  pool->set_kernel_size(4);
  pool->set_stride(2);

  // 4x10
  layer=proto.add_layer();
  layer->set_name("fc1");
  layer->set_type("InnerProduct");
  layer->add_srclayers("pool1");
  InnerProductProto *inner=layer->mutable_inner_product_param();
  inner->set_num_output(10);

  // 4x10
  layer=proto.add_layer();
  layer->set_name("loss");
  layer->set_type("SoftmaxLoss");
  layer->add_srclayers("fc1");
  layer->add_srclayers("label");

  return proto;
}

TEST(NeuralNetTest, NoPartition){
  NetProto proto=CreateNetProto();
  NeuralNet net(proto);
}

TEST(NeuralNetTest, DataPartition){
  NetProto proto=CreateNetProto();
  proto.set_partition_type(kDataPartition);
  NeuralNet net(proto, 2);
}
TEST(NeuralNetTest, ModelPartition){


}
TEST(NeuralNetTest, HyridPartition){


}
