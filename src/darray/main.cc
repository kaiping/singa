#include <iostream>

#include <macdecls.h>
#include <stdio.h>
#include <mpi.h>
#include <glog/logging.h>
#include <ga.h>


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
    LOG(ERROR)<<"before sync";
    DArray::sync();
    LOG(ERROR)<<"after sync";
    if(GArray::Mid == 0)
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
        LOG(ERROR)<<"before Put";
        garr1.Put(larr1);
        garr2.Put(larr2);
        LOG(ERROR)<<"after Put";
        larr1.DeleteStore();
        larr2.DeleteStore();
    }
    DArray::sync();
    DArray garr4 = garr1[4];
    LOG(ERROR)<<"before mult garr4 dim "<<garr4.dim();
    garr3.matrixMult(garr4,garr2);
    LOG(ERROR)<<"after mult";
    DArray larr3 = garr3.Fetch();
    LOG(ERROR)<<"after fetch";
    DArray::sync();
    for(int i = 0; i < 10; i++)
    {
        for(int j = 0; j < 10; j++)
            std::cout<<larr3.v(i,j)<<' ';
        std::cout<<std::endl;
    }
    garr1.DeleteStore();
    garr2.DeleteStore();
    garr3.DeleteStore();
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
void Debug() {
  int i = 0;
  char hostname[256];
  gethostname(hostname, sizeof(hostname));
  printf("PID %d on %s ready for attach\n", getpid(), hostname);
  fflush(stdout);
  while (0 == i)
    sleep(5);
}
int main(int argc, char** argv)
{
  //Range::test();
  //Shape::test();
  //Area::test();
  //DArray::test();
  google::InitGoogleLogging(argv[0]);
  MPI_Init(&argc, &argv);
  GArray::init();
 // Debug();
  MatrixTestG();
  LOG(ERROR)<<"after test g";
  MatrixTestL();
  LOG(ERROR)<<"after test l";
  MatrixTestB();
  LOG(ERROR)<<"after test b";
  GA_Terminate();
  LOG(ERROR)<<"after terminate ";
  MPI_Finalize();
  LOG(ERROR)<<"after mpi ";

  return 0;
}
