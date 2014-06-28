// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-06-28 14:56

#ifndef INCLUDE_UTILS_START_DEAMON_H_
#define INCLUDE_UTILS_START_DEAMON_H_

// the deamon process reads system configuration file, then
// starts coordinator, worker threads,
// and distributed_memory, distributed_disk thread.
int startDeamon(const char* system_conf_path, const char* model_conf_path);

#endif  // INCLUDE_UTILS_START_DEAMON_H_
