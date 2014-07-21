// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-18 19:44
#include <gtest/gtest.h>
#include "proto/lapis.pb.h"
#include "model/blob.h"

namespace lapis {
class BlobTest : public ::testing::Test {
 public:
  BlobTest() : blob1(new Blob()), blob2(new Blob()) {}
  ~BlobTest() {
    delete blob1;
    delete blob2;
  }
 protected:
  Blob *blob1, *blob2;
  Blob blob3, blob4;
};

TEST_F(BlobTest, Constructor) {
  EXPECT_EQ(blob1->Length(), 0);
  EXPECT_EQ(blob1->width(), 0);
  EXPECT_EQ(blob1->height(), 0);

  EXPECT_EQ(blob3.Length(), 0);
  EXPECT_EQ(blob3.width(), 0);
  EXPECT_EQ(blob3.height(), 0);

  EXPECT_TRUE(blob2->content() == nullptr);
  EXPECT_TRUE(blob4.content() == nullptr);
}

TEST_F(BlobTest, TestReshape) {
  blob1->Reshape(10);
  EXPECT_EQ(blob1->Length(), 10);
  EXPECT_EQ(blob1->size(), 1);
  EXPECT_EQ(blob1->height(), 1);
  EXPECT_EQ(blob1->width(), 10);
  EXPECT_TRUE(blob1->content() != nullptr);

  blob2->Reshape(3, 4);
  EXPECT_EQ(blob2->Length(), 12);
  EXPECT_EQ(blob2->size(), 1);
  EXPECT_EQ(blob2->height(), 3);
  EXPECT_EQ(blob2->width(), 4);
  EXPECT_TRUE(blob2->content() != nullptr);

  blob3.Reshape(3, 4, 5);
  EXPECT_EQ(blob3.Length(), 60);
  EXPECT_EQ(blob3.size(), 3);
  EXPECT_EQ(blob3.height(), 4);
  EXPECT_EQ(blob3.width(), 5);
  EXPECT_TRUE(blob3.content() != nullptr);

  blob4.Reshape(3, 4, 5, 6);
  EXPECT_EQ(blob4.Length(), 360);
  EXPECT_EQ(blob4.size(), 3);
  EXPECT_EQ(blob4.height(), 5);
  EXPECT_EQ(blob4.width(), 6);
  EXPECT_TRUE(blob4.content() != nullptr);
}

}  // namespace lapis
