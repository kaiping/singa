#ifndef INCLUDE_SERVER_H_
#define INCLUDE_SERVER_H_

#include "utils/network_service.h"
#include "core/global-table.h"
#include "proto/model.pb.h"
namespace singa {
/**
 * \file server.h defines the Class \class TableServer and the interface of
 * its handlers. Two classes implemented this interface are also provided,
 *  namely, \class TSHandlerForSGD and \class TSHandlerForAda.
 */

/**
 * TableServer runs a loop to handle requests from workers for a table.
 * There are three requests, Put, Get and Update. Every Table is associated
 * with a the \class Tableserverhandler (e.g., \class TSHandlerForSGD).
 */
class TableServer {
public:
	/**
	 * Start the table server. There are several steps:
	 * 1. Create the table of type <TKey, TVal>
	 * 2. Init and start NetworkService.
	 * 3. Register callback for handling requests.
	 * 4. Start the dispatch loop.
	 */
	void Start(const SGDProto & sgd);


	/**
	 * Convert message to PutRequest and invoke table's operation to insert TableData
	 * object to the table.
	 */
	bool handle_put_request(Message *msg);

	/**
	 * Convert message to GetRequest and invoke table's operation to return TableData object.
	 */
	bool handle_get_request(Message *msg);

	/**
	 * Convert message to UpdateRequest and invoke table's operation to update the table.
	 */
	bool handle_update_request(Message *msg);

	/**
	 * Stop the dispatch loop in the main thread. Exit MPI.
	 */
	void handle_shutdown();
private:
	NetworkService *network_service_;
	GlobalTable *table_;
	RequestDispatcher *dispatcher_;

	void create_table(const SGDProto &sgd);
};

/**
 * Table server handler for SGD algorithm.
 * The update considers momentum, learning rate and weight decay.
 */
class TSHandlerForSGD: public TableServerHandler {
 public:
  virtual void Setup(const SGDProto& sgd);
  virtual bool Update(TVal* origin, const TVal& update);

 protected:
  float GetLearningRate(int step, float multiplier){
    float lr=UpdateHyperParam(
        step, learning_rate_change_,
        learning_rate_change_steps_, learning_rate_, gamma_);
    return lr*multiplier;
  }

  float GetWeightDecay(int step, float multiplier){
    return weight_decay_*multiplier;
  }

  float GetMomentum(int step, float multiplier){
    return momentum_;
  }

  float UpdateHyperParam(
      int step, SGDProto::ChangeProto change,
      int change_steps, float a, float b);

 protected:
   float learning_rate_, momentum_, weight_decay_, gamma_;
   int learning_rate_change_steps_;
   SGDProto_ChangeProto learning_rate_change_;
};

/**
 * Table server handler for AdaGrad SGD.
 */
class TSHandlerForAda: public TableServerHandler {
 public:
  virtual void Setup(const SGDProto& sgd);
  virtual bool Update(TVal* origin, const TVal& update);
  virtual bool Put(const TKey& key, TVal* to, const TVal& from);

 protected:
   float learning_rate_;
};

#endif
