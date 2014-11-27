//  Copyright © 2014 Anh Dinh. All Rights Reserved.
#ifndef INCLUDE_CORE_MEMORY_SERVER_H_
#define INCLUDE_CORE_MEMORY_SERVER_H_

#include "core/common.h"
#include "core/table.h"
#include "core/global-table.h"
#include "core/local-table.h"
#include "proto/worker.pb.h"
#include "utils/network_thread.h"

/**
 * @file table_server.h
 * Process table requests. It contains callback implementation for multiple request types.
 */
namespace lapis {

/**
 * The class that processes put/get/update requests to the in-memory tables.
 */
class TableServer: private boost::noncopyable {
public:
	TableServer(){}
	~TableServer() {}

	/**
	 * Initialize the class with the newly created tables. The tables contain no data.
	 *
	 * @param tables int->GlobalTable map created by the main thread.
	 */
	void StartTableServer(const std::map<int, GlobalTable*> &tables);

	/**
	 * Shutdown the service gracefully. Make sure the table checkpointing is completed
	 * successfully (if checkpointing is enabled).
	 */
	void ShutdownTableServer();

	/**
	 * Process remote put request. Simply invoke ApplyPut on the requested table.
	 * Always return true.
	 */
	bool HandlePutRequest(const Message *message);

	/**
	 * Process remote update requests. Invoke ApplyPut on the requested table, which
	 * may return false indicating that the update cannot be applied and the
	 * request must be re-processed later.
	 */
	bool HandleUpdateRequest(const Message *message);

	/**
	 * Process remote get request. Invoke HandleGet on the requested table, which
	 * may return false indicating the request cannot be fulfilled and must be
	 * re-processed later.
	 */
	bool HandleGetRequest(const Message *message);

private:
	int server_id_; /**< the process MPI rank */
	std::shared_ptr<NetworkThread> net_; /**< NetworkThread */
	std::map<int, GlobalTable*> tables_; /**< tables */
};
}  //  namespace lapis

#endif //  INCLUDE_CORE_MEMORY_SERVER_H_
