#include <iostream>

#include <stdio.h>
#include "dalib.h"
//#include "range.h"
//#include "shape.h"
//#include "area.h"
#include "darray.h"

using namespace std;

void MatrixTestG()
{
    std::vector<int> a(3,10);
    std::vector<int> a2(2,10);
    Shape b(a);
    Shape b2(a2);
    DArray garr1 = DArray::Global(b);
    DArray garr2 = DArray::Global(b2);
    DArray garr3 = DArray::Global(b2);
    DArray::sync();
    if(GArray::Mid == 1)
    {
        DArray larr1 = DArray::Local(b);
        DArray larr2 = DArray::Local(b2);
        for(int i = 0; i < 10; i++)
        for(int j = 0; j < 10; j++)
        for(int k = 0; k < 10; k++)
            larr1.v(i,j,k) = i+j-k;
        for(int i = 0; i < 10; i++)
        for(int j = 0; j < 10; j++)
            larr2.v(i,j) = i+2*j;
        garr1.Put(larr1);
        //another way:
        garr2.Copy(larr2);
        larr1.DeleteStore();
        larr2.DeleteStore();
    }
    DArray::sync();
    DArray garr4 = garr1[4];
    garr3.matrixMult(garr4,garr2);
    DArray larr3 = garr3.Fetch();
    garr1.DeleteStore();
    garr2.DeleteStore();
    garr3.DeleteStore();
    DArray::sync();
    for(int i = 0; i < 10; i++)
    {
        for(int j = 0; j < 10; j++)
            std::cout<<larr3.v(i,j)<<' ';
        std::cout<<std::endl;
    }
    larr3.DeleteStore();
}

void MatrixTestL()
{
    std::vector<int> a(3,10);
    std::vector<int> a2(2,10);
    Shape b(a);
    Shape b2(a2);
    DArray larr1 = DArray::Local(b);
    DArray larr2 = DArray::Local(b2);
    DArray larr3 = DArray::Local(b2);
    //std::cout<<101<<std::endl;
    for(int i = 0; i < 10; i++)
    for(int j = 0; j < 10; j++)
    for(int k = 0; k < 10; k++)
        larr1.v(i,j,k) = i+j-k;
    for(int i = 0; i < 10; i++)
    for(int j = 0; j < 10; j++)
        larr2.v(i,j) = i+2*j;
    //std::cout<<102<<std::endl;
    DArray larr4 = larr1[4];
    //std::cout<<larr4.dim()<<std::endl;
    larr3.matrixMult(larr4,larr2);
    //std::cout<<103<<std::endl;
    for(int i = 0; i < 10; i++)
    {
        for(int j = 0; j < 10; j++)
            std::cout<<larr3.v(i,j)<<' ';
        std::cout<<std::endl;
    }
    larr1.DeleteStore();
    larr2.DeleteStore();
    larr3.DeleteStore();
}

void MatrixTestB()
{
    int larr1[10][10][10] = {};
    int larr2[10][10] = {};
    int larr3[10][10] = {};
    //std::cout<<101<<std::endl;
    for(int i = 0; i < 10; i++)
    for(int j = 0; j < 10; j++)
    for(int k = 0; k < 10; k++)
        larr1[i][j][k] = i+j-k;
    for(int i = 0; i < 10; i++)
    for(int j = 0; j < 10; j++)
        larr2[i][j] = i+2*j;
    //std::cout<<102<<std::endl;
    int larr4[10][10] = {};
    for(int i = 0; i < 10; i++)
    for(int j = 0; j < 10; j++)
    larr4[i][j] = larr1[4][i][j];
    //std::cout<<larr4.dim()<<std::endl;
    for(int i = 0; i < 10; i++)
    for(int j = 0; j < 10; j++)
    {
        larr3[i][j] = 0;
        for(int k = 0; k < 10; k++)
            larr3[i][j] += larr4[i][k]*larr2[k][j];
    }
    for(int i = 0; i < 10; i++)
    {
        for(int j = 0; j < 10; j++)
            std::cout<<larr3[i][j]<<' ';
        std::cout<<std::endl;
    }
}

int main()
{
    //Range::test();
    //Shape::test();
    //Area::test();
    //DArray::test();
    //MatrixTestG();
    MatrixTestL();
    MatrixTestB();
    return 0;
}
