//  Copyright © 2014 Anh Dinh. All Rights Reserved.

//  distributed memory interface, exposed to higher application level
//  singleton (similar to NetworkThread)

#ifndef INCLUDE_CORE_DISTRIBUTED-MEMORY_H_
#define INCLUDE_CORE_DISTRIBUTED-MEMORY_H_

#include "table-registry.h"

namespace lapis{
	class DistributedMemory : private boost::noncopyable{
	 public:
		void Init();

		template<class K, class V>
		TypedGlobalTable<K, V>* CreateTable(int id, const GlobalContext& context);

		void AssignTables();  //  assign tables to clients

		static DistributedMemory* Get();

	 private:
		//assign which worker owning this (table,shard)
		void assign_worker(int table, int shard);

		void send_table_assignments();

		bool is_client_started_ = false;
		bool is_initialized = false;

		NetworkThread* net_;
		DistributedMemory();
	};

}  //  namespace lapis

#endif  //  INCLUDE_CORE_DISTRIBUTED-MEMORY_H_
