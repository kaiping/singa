/*
to do list
4 : garray related function
5 : linking
6 : debuging
*/


//
#include <vector>
#include "garray.h"

//int NGA_Create(int type, int ndim, int dims[], char *array_name, int chunk[])

GArray::GArray(const Shape& shape, int mode):
data_(0),dim_(shape.dim()), myshape_(shape),local_(shape),isvalid_(true)
{

    int *dims = new int[dim_];
    int *chunks = new int[dim_];
    for(int i = 0; i < dim_; i++)
    {
        dims[i] = shape[i];
        chunks[i] = PM();
    }
    if(mode == 1)
    {
        chunks[0] = 1;
        chunks[1] = dims[1];
        if(dadebugmode && dim_ != 2)
            errorReport(_CFUNC,"not 2dim for comm array");
    }
    errorReport(_CFUNC,"need to be replaced by comments");//data_ = NGA_Create(C_FLT, dim_, dims, char *array_name, chunks)
    int *lo = new int[dim_];
    int *hi = new int[dim_];
    errorReport(_CFUNC,"need to be replaced by comments");//NGA_Distribution(data_, Mid(), lo, hi);
    std::vector<Range> vshape;
    for(int i = 0; i < dim_; i++)
    {
        if(lo[i] < 0)lo[i] = 0;
        hi[i]++;
        if(hi[i] < 0)hi[i] = 0;
        vshape.push_back(Range(lo[i],hi[i]));
    }
    local_ = Area(vshape);
    //errorReport(_CFUNC,"need to be replaced");
}

void GArray::DeleteStore()
{
    bool isvalid_ = false;
    errorReport(_CFUNC,"need to be replaced by comments");//GA_Destroy(data_);
}

