// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-16 15:06

#include <glog/logging.h>
#include <gtest/gtest.h>
int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::google::InitGoogleLogging(argv[0]);

  return RUN_ALL_TESTS();
}
