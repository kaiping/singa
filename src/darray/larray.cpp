#include "larray.h"

LArray::LArray(const Shape& shape):
myshape_(shape),dim_(shape.dim()),data_(shape.size()),isvalid_(1)
{}


float& LArray::v(const std::vector<int>& index)
{
    if(!isvalid())
        errorReport(_CFUNC,"operating on an invalid array!");
    int iindex = 0;
    for(int i = 0; i < myshape_.dim(); i++)
    {
        iindex *= myshape_[i];
        iindex += index[i];
    }
    if(dadebugmode && iindex >= data_.size())
            errorReport(_CFUNC,"arg too large");
    if(dadebugmode && iindex < 0)
            errorReport(_CFUNC,"arg too small");
    return data_[iindex];
}

const float& LArray::v(const std::vector<int>& index)const
{
    if(!isvalid())
        errorReport(_CFUNC,"operating on an invalid array!");
    int iindex = 0;
    for(int i = 0; i < myshape_.dim(); i++)
    {
        iindex *= myshape_[i];
        iindex += index[i];
    }
    if(dadebugmode && iindex >= data_.size())
            errorReport(_CFUNC,"arg too large");
    if(dadebugmode && iindex < 0)
            errorReport(_CFUNC,"arg too small");
    return data_[iindex];
}

float& LArray::vnext(std::vector<int>& pointer,const Area& area)
{
    float & res = v(pointer);
    int dim = pointer.size()-1;
    while(1)
    {
        if(dim<0)break;
        pointer[dim]+=1;
        if(pointer[dim] < area[dim].end())break;
        pointer[dim] = area[dim].start();
        dim--;
    }
    return res;
}

const float& LArray::vnext(std::vector<int>& pointer,const Area& area)const
{
    const float & res = v(pointer);
    int dim = pointer.size()-1;
    while(1)
    {
        if(dim<0)break;
        pointer[dim]+=1;
        if(pointer[dim] < area[dim].end())break;
        pointer[dim] = area[dim].start();
        dim--;
    }
    return res;
}


std::vector<int> LArray::vfirst(const Area& area)const
{
    std::vector<int> res;
    for(int i = 0; i < area.dim(); i++)
    {
        res.push_back(area[i].start());
    }
    return res;
}

void LArray::DeleteStore()
{
    isvalid_ = false;
    data_.resize(0);
}

void LArray::Map(const LArray& src,float(*mapfunc)(float),const Area& areadst,const Area& areasrc)
{
    if(dadebugmode && areadst.size() !=  areasrc.size())
            errorReport(_CFUNC,"not equal size");
    std::vector<int> dstp = vfirst(areadst);
    std::vector<int> srcp = src.vfirst(areasrc);
    int mysize = areadst.size();
    for(int i = 0; i < mysize; i++)
    {
        vnext(dstp,areadst) = mapfunc(src.vnext(srcp,areasrc));
    }
    return;
}

void LArray::Map(const LArray& src,float value, float(*mapfunc)(float,float),const Area& areadst,const Area& areasrc)
{
    if(dadebugmode && areadst.size() !=  areasrc.size())
            errorReport(_CFUNC,"not equal size");
    std::vector<int> dstp = vfirst(areadst);
    std::vector<int> srcp = src.vfirst(areasrc);
    int mysize = areadst.size();
    for(int i = 0; i < mysize; i++)
    {
        vnext(dstp,areadst) = mapfunc(src.vnext(srcp,areasrc), value);
    }
    return;
}

void LArray::Map(const LArray& src1 ,const LArray& src2,float(*mapfunc)(float,float),const Area& areadst,const Area& areasrc1,const Area& areasrc2)
{
    if(dadebugmode && areadst.size() !=  areasrc1.size())
            errorReport(_CFUNC,"not equal size src1");
    if(dadebugmode && areadst.size() !=  areasrc2.size())
            errorReport(_CFUNC,"not equal size src2");
    std::vector<int> dstp = vfirst(areadst);
    std::vector<int> src1p = src1.vfirst(areasrc1);
    std::vector<int> src2p = src2.vfirst(areasrc2);
    int mysize = areadst.size();
    for(int i = 0; i < mysize; i++)
    {
        float temp1 = src1.vnext(src1p,areasrc1);
        float temp2 = src2.vnext(src2p,areasrc2);
        vnext(dstp,areadst) = mapfunc(temp1, temp2);
    }
    return;
}



void LArray::Max(const LArray& src1 ,const LArray& src2,const Area& areadst,const Area& areasrc1,const Area& areasrc2)
{
    Map(src1,src2,damax,areadst,areasrc1,areasrc2);
    return;
}

void LArray::Min(const LArray& src1 ,const LArray& src2,const Area& areadst,const Area& areasrc1,const Area& areasrc2)
{
    Map(src1,src2,damin,areadst,areasrc1,areasrc2);
    return;
}

void LArray::Max(const LArray& src,float value,const Area& areadst,const Area& areasrc)
{
    Map(src,value,damax,areadst,areasrc);
    return;
}

void LArray::Min(const LArray& src,float value,const Area& areadst,const Area& areasrc)
{
    Map(src,value,damin,areadst,areasrc);
    return;
}

void LArray::Add(const LArray& src1 ,const LArray& src2,const Area& areadst,const Area& areasrc1,const Area& areasrc2)
{
    Map(src1,src2,daadd,areadst,areasrc1,areasrc2);
    return;
}

void LArray::Add(const LArray& src,float value,const Area& areadst,const Area& areasrc)
{
    Map(src,value,daadd,areadst,areasrc);
    return;
}

void LArray::Minus(const LArray& src1 ,const LArray& src2,const Area& areadst,const Area& areasrc1,const Area& areasrc2)
{
    Map(src1,src2,daminus,areadst,areasrc1,areasrc2);
    return;
}

void LArray::Minus(const LArray& src,float value,const Area& areadst,const Area& areasrc)
{
    Map(src,value,daminus,areadst,areasrc);
    return;
}

void LArray::Mult(const LArray& src1 ,const LArray& src2,const Area& areadst,const Area& areasrc1,const Area& areasrc2)
{
    Map(src1,src2,damult,areadst,areasrc1,areasrc2);
    return;
}

void LArray::Mult(const LArray& src,float value,const Area& areadst,const Area& areasrc)
{
    Map(src,value,damult,areadst,areasrc);
    return;
}

void LArray::Div(const LArray& src1 ,const LArray& src2,const Area& areadst,const Area& areasrc1,const Area& areasrc2)
{
    Map(src1,src2,dadiv,areadst,areasrc1,areasrc2);
    return;
}

void LArray::Div(const LArray& src,float value,const Area& areadst,const Area& areasrc)
{
    Map(src,value,dadiv,areadst,areasrc);
    return;
}

void LArray::Exp(const LArray& src1 ,const LArray& src2,const Area& areadst,const Area& areasrc1,const Area& areasrc2)
{
    Map(src1,src2,daexp,areadst,areasrc1,areasrc2);
    return;
}

void LArray::Exp(const LArray& src,float value,const Area& areadst,const Area& areasrc)
{
    Map(src,value,daexp,areadst,areasrc);
    return;
}

void LArray::Copy(const LArray& src,const Area& areadst,const Area& areasrc)
{
    Map(src,dacopy,areadst,areasrc);
    return;
}

void LArray::Threshold(const LArray& src,float value,const Area& areadst,const Area& areasrc)
{
    Map(src,value,dath,areadst,areasrc);
    return;
}

void LArray::Square(const LArray& src,const Area& areadst,const Area& areasrc)
{
    Map(src,src,damult,areadst,areasrc,areasrc);
    return;
}


//aggragation for one dim
//the function and base e.g. max: max -inf sum: add 0
float LArray::MapAgg(float(*myfunc)(float,float), float initvalue,const Area& area)const
{
    std::vector<int> dstp = vfirst(area);
    int mysize = area.size();
    for(int i = 0; i < mysize; i++)
    {
        initvalue = myfunc(initvalue,vnext(dstp,area));
    }
    return initvalue;
}

float LArray::Max(const Area& area)const
{
    return MapAgg(damax,-INF,area);
}

float LArray::Sum(const Area& area)const
{
    return MapAgg(daadd,0,area);
}

//adv aggragation
void LArray::sumExcept(LArray& dst, int dimindex, const Area& areadst, const Area& areasrc)const
{
    if(dadebugmode && dst.dim() !=  1)
        errorReport(_CFUNC,"not reduced to one dimension");
    if(dadebugmode && areasrc[dimindex].length() != areadst.size())
        errorReport(_CFUNC,"not equal size");
    std::vector<int> dstp = dst.vfirst(areadst);
    for(int i = 0; i < areadst.size(); i++)
    {
        Area temparea = areasrc.resize(i,dimindex);
        dst.vnext(dstp,areadst) = Sum(temparea);
    }
    return;
}

void LArray::addVec(const std::vector<float>& foradd,int dimindex, const Area& areadst, int offset)
{
    if(dadebugmode && areadst[dimindex].length() != foradd.size())
        errorReport(_CFUNC,"not equal size");
    std::vector<int> dstp = vfirst(areadst);
    for(int i = 0; i < foradd.size(); i++)
    {
        Area temparea(areadst.resize(i,dimindex));
        Add(*this, foradd[i+offset],temparea,temparea);
    }
    return;
}

void LArray::Reshape(const Shape& shape)
{
    myshape_ = shape;
    dim_ = shape.dim();
}


//this function need all the three arrays to be two dims
//dst[x][z] = src1[x][y]*src2[y][z]
void LArray::matrixMult(LArray& src1,LArray& src2)
{
    if(dadebugmode && dim() != 2)
        errorReport(_CFUNC,"dst not two dims");
    if(dadebugmode && src1.dim() != 2)
        errorReport(_CFUNC,"src1 not two dims");
    if(dadebugmode && src2.dim() != 2)
        errorReport(_CFUNC,"src2 not two dims");
    int x = myshape()[0];
    int z = myshape()[0];
    int y = src2.myshape()[0];
    if(dadebugmode && src1.myshape()[0] != x)
        errorReport(_CFUNC,"x is not equal");
    if(dadebugmode && src1.myshape()[1] != y)
        errorReport(_CFUNC,"y is not equal");
    if(dadebugmode && src2.myshape()[1] != y)
        errorReport(_CFUNC,"z is not equal");
    float* ldst = loc();
    float* lsrc1 = src1.loc();
    float* lsrc2 = src2.loc();
    //doing matrix mult for dst = src1*src2
    //can be further optimized
    for(int i = 0; i < x; i++)
    {
        for(int j = 0; j < z; j++)
        {
            ldst[i*z+j] = 0;
            for(int k = 0; k < y; k++)
            {
                ldst[i*z+j] += lsrc1[i*y+k]*lsrc2[k*z+j];
            }
        }
    }
    //cblas version
    //cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,x,z,y,1.0,lsrc1,y,lsrc2,z,0.0,ldst,z);
    //matrix mult finished
}

//tbd
/*static void LArray::test()
{

}*/



