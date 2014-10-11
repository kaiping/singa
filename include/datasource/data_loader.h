// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-10-11 17:24

#ifndef INCLUDE_DATASOURCE_DATA_LOADER_H_
#define INCLUDE_DATASOURCE_DATA_LOADER_H_
#include <map>
#include <utility>
#include <string>
#include "utils/network_thread.h"
#include "utils/common.h"
#include "proto/model.pb.h"
#include "proto/system.pb.h"

/**
 * Load data from local disk to worker groups.
 * Training data is evenly partitioned onto all worker groups.
 * Validation/Test data is sent to one worker group (i.e., the group 0)
 *
 * Within one group, the data is replciated on all members (workers).
 * If the group expands later, then we just need to copy the data from
 * any member to that new joined member.
 * If the group shrinks, then we delete the data.
 * If group member chagnes, we do shrink and expansion
 */
namespace lapis {
using std::string;
class DataLoader {
 public:
  explicit DataLoader(const ClusterConfig& conf);
  void LoadLocalDataToCluster(const DataSourceProto& ds);
  void LoadDataForPhase(const Phase& phase, const DataSourceProto& ds);
  void RecieveShards();
  void RecieveShardForPhase(const Phase& phase);
  void CopyShardTo(int sid, int dst);
  void DeleteShard(int sid);


  const static string train_shard="train";
  const static string validation_shard="validation";
  const static string test_shard="test";
  const static string shard_log="shard.log";

 private:
  int gid_, rank_;
  std::map<int, std::pair<int, int>> group_range_;
  string folder_;
};
#endif  // INCLUDE_DATASOURCE_DATA_LOADER_H_
}  // namespace lapis
