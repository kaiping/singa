#ifndef LAPIS_UTILS_GLOBAL_CONTEXT
#define LAPIS_UTILS_GLOBAL_CONTEXT

class GlobalContext {
 public:

 private:
  int num_workers; 
  
  //# of nodes working on serving the distributed memory
  int num_memory_servers;

  //# of nodes working on serving the distributed disk
  int num_disk_servers;
};

#endif //LAPIS_UTILS_GLOBAL_CONTEXT 
