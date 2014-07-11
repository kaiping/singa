// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-11 21:54

#include<glog/loggin.h>
#include<gflags/gflags.h>

DEFINE_string(model_conf, "model configuration file");

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

}
