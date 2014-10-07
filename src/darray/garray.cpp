
#include "garray.h"

GArray::GArray(const Shape& shape, int mode):
data_(0),dim_(shape.dim()), myshape_(shape),local_(shape),isvalid_(1)
{
    errorReport(_CFUNC,"need to be replaced");
}
