// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-22 15:34

#include <gtest/gtest.h>
#include <glog/logging.h>
#include "proto/lapis.pb.h"

#include "model/param.h"

namespace lapis {
class ParamTest : public ::testing::Test {
 public:
  ParamTest() {
    wp.set_name("weight");
    wp.add_shape(3);
    wp.add_shape(4);

    bp.set_name("bias");
    bp.add_shape(4);
  }
 protected:
  Param w, b;
  ParamProto wp, bp;
};

TEST_F(ParamTest, ConstantInit) {
  bp.set_init_method(ParamProto::kConstant);
  bp.set_value(0.5);
  b.Init(bp);
  const float * val=b.Content();
  EXPECT_EQ(0.5f, val[0]);
  EXPECT_EQ(0.5f, val[1]);
  EXPECT_EQ(0.5f, val[2]);
  EXPECT_EQ(0.5f, val[3]);

  wp.set_init_method(ParamProto::kConstant);
  wp.set_value(1.5);
  w.Init(wp);
  val=w.Content();
  EXPECT_EQ(1.5f, val[0]);
  EXPECT_EQ(1.5f, val[3]);
  EXPECT_EQ(1.5f, val[4]);
  EXPECT_EQ(1.5f, val[11]);
}

TEST_F(ParamTest, UniformInit) {
  bp.set_init_method(ParamProto::kUniform);
  bp.set_value(1.0f);
  b.Init(bp);
  const float *val=b.Content();
  EXPECT_TRUE(val[0]>=-1 && val[0]<=1);
  EXPECT_TRUE(val[1]>=-1 && val[2]<=1);
  EXPECT_TRUE(val[2]>=-1 && val[2]<=1);
  EXPECT_TRUE(val[3]>=-1 && val[3]<=1);

  wp.set_init_method(ParamProto::kUniform);
  wp.set_value(1.0f);
  w.Init(wp);
  val=w.Content();
  EXPECT_TRUE(val[0]>=-1 && val[0]<=1);
  EXPECT_TRUE(val[3]>=-1 && val[3]<=1);
  EXPECT_TRUE(val[4]>=-1 && val[4]<=1);
  EXPECT_TRUE(val[11]>=-1 && val[11]<=1);
}

TEST_F(ParamTest, UniformSqrtFanInInit) {
  wp.set_init_method(ParamProto::kUniformSqrtFanIn);
  wp.set_value(2.0f);
  w.Init(wp);
  const float *val=w.Content();
  EXPECT_TRUE(val[0]>=-2 && val[0]<=2);
  EXPECT_TRUE(val[3]>=-2 && val[3]<=2);
  EXPECT_TRUE(val[4]>=-2 && val[4]<=2);
  EXPECT_TRUE(val[11]>=-2 && val[11]<=2);
}


TEST_F(ParamTest, UniformSqrtFanInOutInit) {
  wp.set_init_method(ParamProto::kUniformSqrtFanInOut);
  wp.set_value(1.0f);
  wp.set_low(0.f);
  wp.set_high(1.f);
  w.Init(wp);
  const float *val=w.Content();
  LOG(INFO)<<val[0]<<" "<<val[1]<<" "<<val[2]<<" "<<val[3];
  LOG(INFO)<<val[4]<<" "<<val[5]<<" "<<val[6]<<" "<<val[7];
  LOG(INFO)<<val[8]<<" "<<val[9]<<" "<<val[10]<<" "<<val[11];

  float factor=wp.value()/sqrt(wp.shape().Get(0)+wp.shape().Get(1));
  EXPECT_TRUE(val[0]>=0 && val[0]<=factor);
  EXPECT_TRUE(val[3]>=0 && val[3]<=factor);
  EXPECT_TRUE(val[4]>=0 && val[4]<=factor);
  EXPECT_TRUE(val[11]>=0 && val[11]<=factor);
}

TEST_F(ParamTest, GaussianInit) {
  wp.set_init_method(ParamProto::kGaussain);
  wp.set_value(1.0f);
  wp.set_mean(0);
  wp.set_std(1.0f);
  w.Init(wp);
  const float *val=w.Content();
  LOG(INFO)<<val[0]<<" "<<val[1]<<" "<<val[2]<<" "<<val[3];
  LOG(INFO)<<val[4]<<" "<<val[5]<<" "<<val[6]<<" "<<val[7];
  LOG(INFO)<<val[8]<<" "<<val[9]<<" "<<val[10]<<" "<<val[11];
}

TEST_F(ParamTest, GaussianSqrtFanInInit) {
  wp.set_init_method(ParamProto::kGaussainSqrtFanIn);
  wp.set_value(1.0f);
  wp.set_mean(0);
  wp.set_std(1.0f);
  w.Init(wp);
  const float *val=w.Content();
  LOG(INFO)<<val[0]<<" "<<val[1]<<" "<<val[2]<<" "<<val[3];
  LOG(INFO)<<val[4]<<" "<<val[5]<<" "<<val[6]<<" "<<val[7];
  LOG(INFO)<<val[8]<<" "<<val[9]<<" "<<val[10]<<" "<<val[11];
}
}  // namespace lapis


