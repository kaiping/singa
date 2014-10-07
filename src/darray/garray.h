#ifndef GARRAY_H_INCLUDED
#define GARRAY_H_INCLUDED

#include<iostream>
#include<vector>
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
    GArray(const Shape&,int);
    inline int dim(){return dim_;};
    Area local(){return local_;};
    void DeleteStore(){};
    bool isvalid()const{return isvalid_;};
    LArray Fetch(LArray&,const Area&)const;
    LArray FetchLocal(const Area&)const;
    LArray FetchLocal()const;
    void Put(const LArray&,const Area&, const Area&);
    void PutLocal(const LArray&,const Area&, const Area&);
    void PutLocal(const LArray&,const Area&);
};



#endif //GARRAY_H_INCLUDED
