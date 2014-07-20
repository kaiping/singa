// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-20 11:51

#include <gtest/gtest.h>
#include <glog/logging.h>

#include <fstream> // #NOLINT
#include <string>
#include <vector>

#include "proto/lapis.pb.h"
#include "disk/feature_reader.h"

namespace lapis {
class FeatureReaderTest : public ::testing::Test {
 public:
  FeatureReaderTest() {
    DataSourceProto ds;
    std::string path = "src/test/data/feature_reader.dat";
    ds.set_path(path);
    ds.set_width(5);

    std::ofstream out(path, std::ofstream::binary | std::ofstream::out);
    // note if the precision (6) of out is not changed, then 100000.1f will be
    // 100000 in the file
    out << 0.1f << " " << 1.1f << " " <<4.1f << " " << 0.0f << " "<< 100.1f
        << std::endl;
    out << " " << 0.4f << " " << 1.4f << " " << 4.4f << " " << 0.0f << " " <<
        100.4f;
    out << " " << -0.4f << " " << -1.4f << " " << -4.4f << " " << 0.0f << " "
        << -100.4f;
    out.close();

    std::vector<std::string> vec;;
    reader.Init(ds, vec, 0);
  }
 protected:
  FeatureReader reader;
};

//! check the correctness of the data read by ReadNextRecord
TEST_F(FeatureReaderTest, ReadNextRecord) {
  std::string key;
  float *val = new float[5];

  reader.ReadNextRecord(&key, val);
  EXPECT_EQ(0.1f, val[0]);
  EXPECT_EQ(1.1f, val[1]);
  EXPECT_EQ(4.1f, val[2]);
  EXPECT_EQ(0.0f, val[3]);
  EXPECT_EQ(100.1f, val[4]);

  reader.ReadNextRecord(&key, val);
  reader.ReadNextRecord(&key, val);
  EXPECT_EQ(-0.4f, val[0]);
  EXPECT_EQ(-1.4f, val[1]);
  EXPECT_EQ(-4.4f, val[2]);
  EXPECT_EQ(0.0f, val[3]);
  EXPECT_EQ(-100.4f, val[4]);
}

TEST_F(FeatureReaderTest, EndofFile) {
  std::string key;
  float *val = new float[5];
  bool status;
  status = reader.ReadNextRecord(&key, val);
  EXPECT_TRUE(status);
  status = reader.ReadNextRecord(&key, val);
  EXPECT_TRUE(status);
  status = reader.ReadNextRecord(&key, val);
  EXPECT_TRUE(status);
  status = reader.ReadNextRecord(&key, val);
  EXPECT_FALSE(status);
}

}  // namespace lapis
