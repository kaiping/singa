#ifndef DALIB_H_INCLUDED
#define DALIB_H_INCLUDED

#define name_to_str(xxx) #xxx
#define _CFUNC __PRETTY_FUNCTION__
#include <string>
#include <cmath>

const int dadebugmode = 1;
// 0 for release mode : little check will be made, no error will be reported
// 1 for debug mode : will report error, un-optimized operation will be done anyway with no warning
// 2 for debug mode : will report error, will report warning for un-optimized operations

const float INF = 1e+038;

inline int PM()
{
    return 0;
}


inline void errorReport(const char* name, const char* mode = "")
{
    std::cout<<"Error: "<<name<<std::endl;
    if(mode != "")std::cout<<"Mode: "<<mode<<std::endl;
    return;
}

inline void warningReport(const char* name, const char* mode = "")
{
    std::cout<<"Warning: "<<name<<std::endl;
    if(mode != "")std::cout<<"Mode: "<<mode<<std::endl;
    return;
}

inline void debugReport(const char* name, int mode)
{
    std::cout<<"debug: "<<name<<std::endl;
    std::cout<<mode<<std::endl;
    return;
}

inline void debugReport(const char* name)
{
    std::cout<<"debug: "<<name<<std::endl;
    return;
}

inline void teststart(const char* name)
{
    std::cout<<"Testing: "<<name<<std::endl;
    return;
}

inline void testend(const char* name)
{
    std::cout<<"End of test: "<<name<<std::endl;
    return;
}

inline int damax(int a, int b){return a>b?a:b;}
inline int damin(int a, int b){return a<b?a:b;}
inline int daadd(int a, int b){return a+b;}
inline int daminus(int a, int b){return a-b;}
inline int damult(int a, int b){return a*b;}
inline int dadiv(int a, int b){return a/b;}
inline int daexp(int a, int b){return (int)pow(a,b);}
inline bool dacomp(int a, int b){return a<b;}


inline float damax(float a, float b){return a>b?a:b;}
inline float damin(float a, float b){return a<b?a:b;}
inline float daadd(float a, float b){return a+b;}
inline float daminus(float a, float b){return a-b;}
inline float damult(float a, float b){return a*b;}
inline float dadiv(float a, float b){return a/b;}
inline float daexp(float a, float b){return pow(a,b);}
inline bool dacomp(float a, float b){return a<b;}
inline float dath(float a, float b)
{
    if(dacomp(a,b))return 0;
    else return 1;
}
inline float dacopy(float a){return a;}

#endif //DALIB_H_INCLUDED
