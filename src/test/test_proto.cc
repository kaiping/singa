// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-15 21:54

#include <gtest/gtest.h>
#include "proto/lapis.pb.h"
#include "utils/proto_helper.h"

TEST(ProtoTest, ReadFromFile) {
  ModelConfProto model;
  lapis::ReadProtoFromTextFile("data/model.conf", &model);
  ASSERT_STREQ("simple_autoencoder", net.name());

  NetProto net=model.net();
  ASSERT_EQ(3, net.layer().size());
  ASSERT_EQ(3, net.edge().size());

  LayerProto layer1=net.layer().Get(0);
  ASSERT_STREQ("input", layer1.name());
  ASSERT_EQ(3, layer1.num_output());
  ASSERT_STREQ("Data", layer1.type());
  ASSERT_STREQ("input_hidden", layer1.out_edge());

  EdgeProto edge2=net.edge().Get(1);
  ASSERT_STREQ("hidden_reconstruct", edge2.name());
  ASSERT_STREQ("InnerProduct", edge2.type());
  ASSERT_EQ(2, edge2.param().size());

  ParamProto param=edge2.param().Get(0);
  ASSERT_STREQ("Gaussian", param.initializer());
  ASSERT_EQ(2, param.shape().size());
  ASSERT_EQ(2, param.shape().Get(0));

  TrainerProto trainer=model.trainer();
  ASSERT_EQ(2, trainer.checkpoint_after_steps());
  ASSERT_EQ(2, trainer.checkpoint_every_steps());
  SGDProto sgd=trainer.sgd();
  ASSERT_EQ(0.1, sgd.base_learning_rate());
  ASSERT_EQ(0.5, sgd.base_momentum());
  ASSERT_EQ(0.9, sgd.final_momentum());
  ASSERT_EQ(10, sgd.momentum_change_steps());
}
