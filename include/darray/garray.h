#ifndef GARRAY_H_INCLUDED
#define GARRAY_H_INCLUDED

#include<iostream>
#include<vector>
#include <ga.h>
//#include <macdecls.h>


#include "dalib.h"
#include "shape.h"
#include "area.h"
//#include "darray.h"
#include "larray.h"

class GArray{
    private:
    int data_;
    int dim_;
    Shape myshape_;
    Area local_;
    bool isvalid_;

    public:
    static void init();
    GArray(const Shape&,int);
    inline int dim(){return dim_;};
    Area local(){return local_;};
    void DeleteStore();
    bool isvalid()const{return isvalid_;};
    LArray* Fetch(const Area&)const;
    void Put(LArray&,const Area&);

    static int Nmachine;
    static int Mid;
};



#endif //GARRAY_H_INCLUDED
