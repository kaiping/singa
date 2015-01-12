#ifndef INCLUDE_UTILS_SINGLETON_H_
#define INCLUDE_UTILS_SINGLETON_H_
template<typename T>
class Singleton {
 public:
  static T& Instance() {
    if (!data_) {
      data_ = new T();
    }
    return *data_;
  }
 private:
  static T* data_;
};

template<typename T> T* Singleton<T>::data_ = nullptr;

#endif // INCLUDE_UTILS_SINGLETON_H_
