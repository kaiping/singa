//  Copyright © 2014 Anh Dinh. All Rights Reserved.

//  distributed memory interface, exposed to higher application level
//  singleton (similar to NetworkThread)

#ifndef INCLUDE_CORE_DISTRIBUTED-MEMORY_H_
#define INCLUDE_CORE_DISTRIBUTED-MEMORY_H_

#include "table-registry.h"

namespace lapis{
	class DistributedMemory : private boost::noncopyable{
	 public:
		template<class K, class V>
		TypedGlobalTable<K, V>* CreateTable(int id, int shards, const GlobalContext& context);

		void AssignTables();  //  assign tables to clients

		static DistributedMemory* Get();

	 private:
		bool is_client_started_ = false;
		DistributedMemory();
	};

}  //  namespace lapis

#endif  //  INCLUDE_CORE_DISTRIBUTED-MEMORY_H_
