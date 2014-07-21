// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-20 19:10
#include <gtest/gtest.h>
#include <glog/logging.h>
#include "proto/lapis.pb.h"
#include "disk/idx_reader.h"

namespace lapis {
class IDXReaderTest : public ::testing::Test {
 public:
  IDXReaderTest() {
    DataSourceProto ds;
    // ds.set_path("src/test/data/idx_reader.dat");
    // TODO(wangwei) change the test file to a smaller one
    ds.set_path("examples/mnist/t10k-images-idx3-ubyte");
    ds.set_height(28);
    ds.set_width(28);
    ds.set_size(10000);
    std::vector<std::string> vec;
    reader.Init(ds, vec, 0);
  }

 protected:
  IDXReader reader;
};


TEST_F(IDXReaderTest, ReadNextRecord) {
  std::string key;
  float *val = new float[28 * 28];
  reader.ReadNextRecord(&key, val);
  // the expected data is from Theano tutorial which preprocessed it by /256
  EXPECT_EQ(0.328125f, val[202] / 256);
  EXPECT_EQ(0.0703125, val[741] / 256);
  int offset = reader.Offset();
  reader.ReadNextRecord(&key, val);
  EXPECT_EQ(0.453125, val[94] / 256);
  EXPECT_EQ(0.16015625, val[636] / 256);
  int len = reader.Offset() - offset;
  // the data is of type unsigned char for mnist dataset, no need to *4
  EXPECT_EQ(28 * 28, len);
}

}  // namespace lapis




