// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-11-29 13:46

#ifndef INCLUDE_SERVER_H_
#define INCLUDE_SERVER_H_

#include "core/sparse-table.h"
#include "proto/model.pb.h"
namespace lapis {
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
	bool handle_put_request(const Message *msg);

	/**
	 * Convert message to GetRequest and invoke table's operation to return TableData object.
	 */
	bool handle_get_request(const Message *msg);

	/**
	 * Convert message to UpdateRequest and invoke table's operation to update the table.
	 */
	bool handle_update_request(const Message *msg);

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
 * Base class, specifies the interface of request handlers of table server.
 */
class TableServerHandler: public BaseUpdateHandler<TKey, TVal>{
 public:
  virtual void Setup(const SGDProto& sgd);
  virtual bool CheckpointNow(const TKey& key, const TVal& val);

  virtual bool Update(TVal* origin, const TVal& update)=0;
  virtual bool Get(const TKey& key, const TVal &from, TVal* to);
  virtual bool Put(const TKey& key, TVal* to, const TVal& from);

 protected:
  int checkpoint_after_, checkpoint_frequency_;
  bool synchronous_;
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

/****************************************************************************/
/**
 * Register Tableserverhandler with type identifier
 * @param type identifier of the Tableserverhandler e.g., TSHandlerForSGD
 * @param handler the child layer class
 */
#define REGISTER_TSHandler(type, handler) TSHandlerFactory::Instance()->\
  RegisterCreateFunction(type,\
                        [](void)-> TableServerHandler* {return new handler();})

/**
 * Factory for creating Tableserverhandler based on user provided type string.
 * Users are required to register user-defined Tableserverhandler before
 * creating instances of them during runtime. For example, if you define a
 * new Tableserverhandler TSHandlerForFoo with identifier "Foo", then you can
 * use it by 1) register it (e.g., at the start of the program); 2)Then call
 * TSHandlerFactory::Instance()->Create("Foo") to create an instance.
 */
class TSHandlerFactory {
 public:
  /**
   * static method to get instance of this factory
   */
  static std::shared_ptr<TSHandlerFactory> Get();
  /**
   * Register user defined handler.
   * It adds the handler type/identifier and a function which creats an
   * instance of this. This function is called by the REGISTER_TSHandler macro.
   * @param id identifier of the handler
   * @param create_function a function that creates a handler instance
   */
  void RegisterCreateFunction(
      const std::string id,
      std::function<TableServerHandler*(void)> create_function);
  /**
   * create a layer  instance by providing its type
   * @param type the identifier of the layer to be created
   */
  TableServerHandler *Create(const std::string id);

 private:
  //! To avoid creating multiple instances of this factory in the program
  TSHandlerFactory();
  //! Map that stores the registered handlers
  std::map<std::string, std::function<TableServerHandler*(void)>> map_;
  static std::shared_ptr<TSHandlerFactory> instance_;
};


} /* lapis */

#endif
