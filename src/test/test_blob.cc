// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-18 19:44
#include <gtest/gtest.h>
#include "proto/lapis.pb.h"
#include "model/blob.h"

namespace lapis {
class BlobTest : public ::testing::Test {
 public:
  BlobTest() : blob1(new Blob()), blob2(new Blob()) {}
  ~BlobTest() {delete blob1; delete blob2;}
 private:
  Blob *blob1, *blob2;
  Blob blob3, blob4;
};

TEST_F(BlobTest, Constructor) {
  EXPECT_EQ(blob1->Size(), 0);
  EXPECT_EQ(blob1->Width(), 0);
  EXPECT_EQ(blob1->Height(), 0);

  EXPECT_EQ(blob3.Size(), 0);
  EXPECT_EQ(blob3.Width(), 0);
  EXPECT_EQ(blob3.Height(), 0);

  EXPECT_TRUE(blob2->Content==nullptr);
  EXPECT_TRUE(blob4.Content==nullptr);
}

TEST_F(BlobTest, TestReshape) {
  blob1->Reshape(10);
  EXPECT_EQ(blob1->Size(), 10);
  EXPECT_EQ(blob1->Num(), 1);
  EXPECT_EQ(blob1->Height(), 1);
  EXPECT_EQ(blob1->Width(), 10);
  EXPECT_EQ(blob1->Content()!=nullptr);

  blob2->Reshape(3,4);
  EXPECT_EQ(blob2->Size(), 12);
  EXPECT_EQ(blob2->Num(), 1);
  EXPECT_EQ(blob2->Height(), 3);
  EXPECT_EQ(blob2->Width(), 4);
  EXPECT_TRUE(blob2->Content()!=nullptr);

  blob3.Reshape(3,4,5);
  EXPECT_EQ(blob3.Size(), 60);
  EXPECT_EQ(blob3.Num(), 3);
  EXPECT_EQ(blob3.Height(), 4);
  EXPECT_EQ(blob3.Width(), 5);
  EXPECT_TRUE(blob3.Content()!=nullptr);

  blob4.Reshape(3,4,5,6);
  EXPECT_EQ(blob4.Size(), 360);
  EXPECT_EQ(blob4.Num(), 3);
  EXPECT_EQ(blob4.Height(), 5);
  EXPECT_EQ(blob4.Width(), 6);
  EXPECT_TRUE(blob4.Content()!=nullptr);
}

}
