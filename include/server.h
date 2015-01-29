#ifndef INCLUDE_SERVER_H_
#define INCLUDE_SERVER_H_

#include "utils/network_service.h"
#include "core/global-table.h"
#include "core/common.h"
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
  TableServer();
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
  virtual bool Put(const TKey& key, TVal* to, const TVal& from);

 protected:
  float GetLearningRate(int step, float multiplier){
    float lr=UpdateHyperParam(
        step, sgd_.learning_rate_change(),
        sgd_.learning_rate_change_frequency(),
        sgd_.learning_rate(),
        sgd_.gamma(),
        sgd_.pow());
    return lr*multiplier;
  }

  float GetWeightDecay(int step, float multiplier){
    if(sgd_.has_weight_decay())
      return sgd_.weight_decay()*multiplier;
    else return 0;
  }

  float GetMomentum(int step, float multiplier){
    if(sgd_.has_momentum())
      return sgd_.momentum()*multiplier;
    else return 0;
  }
  float UpdateHyperParam(
      int step, SGDProto::ChangeProto change,
      int change_steps, float a, float b, float c);

 protected:
   SGDProto sgd_;
};

/**
 * Table server handler for AdaGrad SGD.
 * The parameter p, is updated according to p=p-\alpha*g_t/\sqrt(\sum_0^t g_k^2),
 * \alpha is the initial learning rate, g_k is the gradient from k-th step,
 * t is the current step.
 */
class TSHandlerForAda: public TableServerHandler {
 public:
  virtual void Setup(const SGDProto& sgd);
  virtual bool Update(TVal* origin, const TVal& update);
  virtual bool Put(const TKey& key, TVal* to, const TVal& from);

 protected:
   float learning_rate_;
};
}

#endif
