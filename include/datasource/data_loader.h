// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-10-11 17:24

#ifndef INCLUDE_DATASOURCE_DATA_LOADER_H_
#define INCLUDE_DATASOURCE_DATA_LOADER_H_
#include <vector>
#include <utility>
#include <string>
#include <memory>
#include "utils/global_context.h"
#include "utils/common.h"
#include "proto/model.pb.h"

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

const static string train_shard="train";
const static string validation_shard="validation";
const static string test_shard="test";

class DataLoader {
  public:
    DataLoader(const std::shared_ptr<GlobalContext>& gc);
    void ShardData(const DataProto& proto) ;
    void ShardData(const DataSourceProto& source, int ngroups);
    void CreateLocalShard(const DataSourceProto& source, const ShardProto& shard);
    void CreateLocalShards(const DataProto& dp) ;
  private:
    string shard_folder_;
    int gid_, ngroups_, rank_, nprocs_;
};
#endif  // INCLUDE_DATASOURCE_DATA_LOADER_H_
}  // namespace lapis
