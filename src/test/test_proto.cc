// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-15 21:54

#include <gtest/gtest.h>
#include "proto/lapis.pb.h"
#include "utils/proto_helper.h"
namespace lapis {

TEST(ProtoTest, ReadFromFile) {
  ModelConfProto model;
  lapis::ReadProtoFromTextFile("src/test/data/model.conf", &model);
  EXPECT_STREQ("simple_autoencoder", model.name().c_str());
  NetProto net = model.net();
  EXPECT_EQ(3, net.layer().size());
  EXPECT_EQ(3, net.edge().size());
  LayerProto layer1 = net.layer().Get(0);
  EXPECT_STREQ("input", layer1.name().c_str());
  EXPECT_EQ(3, layer1.num_output());
  EXPECT_STREQ("Data", layer1.type().c_str());
  EXPECT_STREQ("input_hidden", layer1.out_edge().Get(0).c_str());
  EdgeProto edge2 = net.edge().Get(1);
  EXPECT_STREQ("hidden_reconstruct", edge2.name().c_str());
  EXPECT_STREQ("InnerProduct", edge2.type().c_str());
  EXPECT_EQ(2, edge2.param().size());
  ParamProto param = edge2.param().Get(0);
  EXPECT_TRUE(ParamProto::kGaussain == param.init_method());
  EXPECT_EQ(2, param.shape().size());
  EXPECT_EQ(2, param.shape().Get(0));
  TrainerProto trainer = model.trainer();
  EXPECT_EQ(2, trainer.checkpoint_after_steps());
  EXPECT_EQ(2, trainer.checkpoint_every_steps());
  SGDProto sgd = trainer.sgd();
  // must write 0.1f instead of 0.1
  EXPECT_EQ(0.1f, sgd.base_learning_rate());
  EXPECT_EQ(0.5f, sgd.base_momentum());
  EXPECT_EQ(0.9f, sgd.final_momentum());
  EXPECT_EQ(10, sgd.momentum_change_steps());
}

}  // namespace lapis
