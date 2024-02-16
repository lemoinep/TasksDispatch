
#include <thread>
#include <vector>
#include <array>
#include <typeinfo>
#include <iostream>
#include <mutex>
#include <sched.h>
#include <pthread.h>

#include<algorithm> //for Each_fors



#include <string>

#include <utility>
#include <functional>

#include <future>

#include <cassert>

#include <chrono>

#include <type_traits>

#include <list>
#include <ranges>

#include "SpDataAccessMode.hpp"
#include "Utils/SpUtils.hpp"

#include "Task/SpTask.hpp"
#include "Legacy/SpRuntime.hpp"
#include "Utils/SpTimer.hpp"
#include "Utils/small_vector.hpp"
#include "Utils/SpConsumerThread.hpp"


#include "napp.hpp"
#include "Tools.hpp"

#include "TasksDispach.hpp"



//================================================================================================================================
// BLOCK TEST Add vectors
//================================================================================================================================

void activeBlockTest_Add_Vectors()
{
    //ADD 2 vectors
    int nbThreads = 4;
    //long int nbN=12;
    long int nbN=4*6;
    int sizeBlock=nbN/nbThreads;
    int diffBlock=nbN-sizeBlock*nbThreads;
    std::vector<int> VecA;
    std::vector<int> VecB;
    for(int i=0;i<nbN;i++) { 
        VecA.push_back(i);  
        //VecB.push_back(nbN-i);    
        VecB.push_back(i);   
    }
    //std::vector<int> VecR;
    std::vector<int> VecR(nbN,0);

    auto MyAlgo000=[VecA,VecB,sizeBlock,&VecR](const int& k) {  
            //std::cout<<"wValue k="<<k<< std::endl;
            int vkBegin=k*sizeBlock;
            int vkEnd=(k+1)*sizeBlock;
            for(int j=vkBegin;j<vkEnd;j++)
            {
                VecR.push_back(VecA[j]+VecB[j]);    
            }
        return true;
    };

    std::cout<<"Calcul with std::async"<<"\n";
    VecR.clear();
    TasksDispach FgCalcul; 
    FgCalcul.init(1,nbThreads,true); FgCalcul.setFileName("TestDispachSum");  FgCalcul.qViewChrono=false;
    FgCalcul.run(MyAlgo000);
    std::cout << "Vec A= "; 
    for (auto it = VecA.begin(); it != VecA.end(); it++) { std::cout << *it << " "; }
    std::cout << "\n"; 
    std::cout << "Vec B= "; 
    for (auto it = VecB.begin(); it != VecB.end(); it++) { std::cout << *it << " "; }
    std::cout << "\n"; 
    std::cout << "Vec R= "; 
    for (int k=0; k<VecR.size(); k++) { Color(k/sizeBlock+1);std::cout << VecR[k] << " "; }
    Color(7);

    std::cout << "\n";
    std::cout << "\n"; 
    std::cout<<"Calcul with Specx"<<"\n";
    VecR.clear();
    FgCalcul.init(2,nbThreads,true); FgCalcul.setFileName("TestDispachSum"); FgCalcul.qViewChrono=false;    //FgCalcul.qSave=false; 
    FgCalcul.run(MyAlgo000);
    std::cout << "Vec A= "; 
    for (auto it = VecA.begin(); it != VecA.end(); it++) { std::cout << *it << " "; }
    std::cout << "\n"; 
    std::cout << "Vec B= "; 
    for (auto it = VecB.begin(); it != VecB.end(); it++) { std::cout << *it << " "; }
    std::cout << "\n"; 
    std::cout << "Vec R= "; 
    for (int k=0; k<VecR.size(); k++) { Color(k/sizeBlock+1);std::cout << VecR[k] << " "; }
    Color(7);
    std::cout << "\n"<< "\n"; 
}



//================================================================================================================================
// BLOCK TEST Matrix Products
//================================================================================================================================

void activeBlockTest_Matrix_Products()
{
    //Matrix product
    int nbThreads = 3; int row=3; int col=3;
    int MatA[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    int MatB[3][3] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    int MatR[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};

    std::cout<<"MatA"<< "\n";
    WriteMat(MatA,row,col);
    std::cout<<"MatB"<< "\n";
    WriteMat(MatB,row,col);
    
     auto MyAlgo000=[MatA,MatB,row,col,&MatR](const int& k) {  
            for(int r=0; r<row; ++r)
            {
                int s=0; for(int c=0; c<col; ++c) { s+=MatA[r][c]*MatB[c][k];}
                MatR[r][k]=s;
            }
        return true;
    };

    TasksDispach FgCalcul1; 
    FgCalcul1.init(1,nbThreads,true); FgCalcul1.setFileName("TestDispachMult"); 
    FgCalcul1.run(MyAlgo000);
    std::cout<<"MatR=MatA*MatB with STD::ASYNC"<< "\n";
    WriteMat(MatR,row,col);

    TasksDispach FgCalcul2; 
    FgCalcul2.init(2,nbThreads,true); FgCalcul2.setFileName("TestDispachMult");
    FgCalcul2.run(MyAlgo000);
    std::cout<<"MatR=MatA*MatB with Specx"<< "\n";
    WriteMat(MatR,row,col);
    std::cout << "\n"<< "\n"; 
}

//================================================================================================================================
// BLOCK TEST Pstimation of pi
//================================================================================================================================

void activeBlockTest_Integral001()
{
    //CALCUL DE PI par les deux méthodes std::async et Specx
    int nbThreads = 6;
    long int nbN=1000000;
    int sizeBlock=nbN/nbThreads;
    int diffBlock=nbN-sizeBlock*nbThreads;
    double h=1.0/double(nbN);
    double integralValue=0.0;
    std::vector<double> valuesVec;
    valuesVec.clear();

    auto MyAlgo000=[h,sizeBlock,&valuesVec](const int& k) {  
            //std::cout<<"wValue k="<<k<< std::endl;
            int vkBegin=k*sizeBlock;
            int vkEnd=(k+1)*sizeBlock;
            double sum=0.0; double x;
            for(int j=vkBegin;j<vkEnd;j++)
            {
                x=h*double(j);
                sum+=4.0/(1.0+x*x);
            }
            valuesVec.push_back(sum);  
            //std::cout<<"Sum "<<sum<<"\n";    
        return true;
    };
   
    TasksDispach FgCalculIntegral; 
    std::cout<<"\n";
    std::cout<<"\n";
    std::cout<<"PI method std::async"<<"\n";
    valuesVec.clear(); std::cout<<"Clear results size="<<valuesVec.size()<< "\n";
    FgCalculIntegral.init(1,nbThreads,true); FgCalculIntegral.setFileName("TestDispachIntegral"); FgCalculIntegral.qViewChrono=false;
    FgCalculIntegral.run(MyAlgo000);
    Color(7);

    std::cout << "\n"; 
    integralValue=h*std::reduce(valuesVec.begin(),valuesVec.end()); 
    std::cout<<"PI Value= "<<integralValue<<"\n";

    /*
    std::cout<<"PI method Specx"<<"\n";
    valuesVec.clear(); std::cout<<"Clear results size="<<valuesVec.size()<< "\n";
    FgCalculIntegral.init(2,nbThreads,true); FgCalculIntegral.setFileName("TestDispachIntegral"); FgCalculIntegral.qViewChrono=false;
    FgCalculIntegral.run(MyAlgo000);
    Color(7);
    std::cout << "\n"; 
    integralValue=h*std::reduce(valuesVec.begin(),valuesVec.end()); 
    std::cout<<"PI Value= "<<integralValue<<"\n"*/
    
}


void activeBlockTest_Integral002(int nbThreads)
{
    long int nbN=1000000;
    int sizeBlock=nbN/nbThreads;
    int diffBlock=nbN-sizeBlock*nbThreads;
    double h=1.0/double(nbN);
    double integralValue=0.0;


    auto MyAlgo000=[h,sizeBlock](const int k,double& s) {  
            int vkBegin=k*sizeBlock;
            int vkEnd=(k+1)*sizeBlock;
            double sum=0.0; double x=0.0;
            for(int j=vkBegin;j<vkEnd;j++) { x=h*double(j); sum+=4.0/(1.0+x*x); }
            s=sum;
            //std::cout <<s<< "\n"; 
        return true;
    };

    std::cout<<"\n";
    std::cout<<"PI method (2) STD::ASYNC"<<"\n";
    TasksDispach Fg1; 
    Fg1.setFileName("Test with STD::ASYNC2"); 
    Fg1.init(1,nbThreads,true);  Fg1.qInfo=false; Fg1.qViewChrono=false; Fg1.qSave=true; Fg1.setFileName("TestDispachIntegral");
    std::vector<double> valuesVec1=Fg1.sub_run_async_beta(MyAlgo000);
    std::cout << "Vec R= "; for (int k=0; k<nbThreads; k++) { Color(k+1); std::cout << valuesVec1[k] << " ";  } 
    std::cout << "\n"; 
    Color(7);
    std::cout << "\n"; 
    integralValue=h*std::reduce(valuesVec1.begin(),valuesVec1.end()); 
    std::cout<<"PI Value= "<<integralValue<<"\n";
    std::cout << "\n"; 


    
    std::cout<<"\n";
    std::cout<<"PI method (2) Specx"<<"\n";
    TasksDispach Fg2; 
    Fg2.setFileName("Test with STD::ASYNC2"); 
    Fg2.init(2,nbThreads,true);  Fg2.qInfo=false; Fg2.qViewChrono=false; Fg2.qSave=true; Fg2.setFileName("TestDispachIntegral");
    std::vector<double> valuesVec2=Fg2.sub_run_specx(MyAlgo000);
    std::cout << "Vec R= "; for (int k=0; k<nbThreads; k++) { Color(k+1); std::cout << valuesVec2[k] << " ";  } 
    std::cout << "\n"; 
    Color(7);
    std::cout << "\n"; 
    integralValue=h*std::reduce(valuesVec2.begin(),valuesVec2.end()); 
    std::cout<<"PI Value= "<<integralValue<<"\n";
    std::cout << "\n"; 
}


void activeBlockTest_Integral003()
{
    int nbThreads = 96;
    double integralValue=0.0;
    double R[nbThreads];
    for (int k=0; k<nbThreads; k++) { R[k]=0.0; }

    auto MyAlgo000=[&R](const int& k) mutable {  
            double sum=0.0; double x;
            for(int j=0;j<10;j++)
            { 
                sum+=double(j);
            }
            usleep(100);
            R[k]=sum+double(k);      
            //std::cout<<"Sum "<<R[k]<<"\n";
        return true;
    };
   
    TasksDispach Fg; 
    Fg.setFileName("TestDispachIntegral"); 
    Fg.init(1,nbThreads,true);  Fg.qInfo=false;
    Fg.run(MyAlgo000);
    std::cout << "Vec R= "; for (int k=0; k<nbThreads; k++) { Color(k+1); std::cout << R[k] << " ";  } 
    std::cout << "\n"; 

    for (int k=0; k<nbThreads; k++) { R[k]=0.0; }
    Fg.init(2,nbThreads,true); Fg.qInfo=false;
    Fg.run(MyAlgo000);

    std::cout << "Vec R= "; for (int k=0; k<nbThreads; k++) { Color(k+1); std::cout << R[k] << " ";  } 
    Color(7);
    std::cout << "\n"; 
}



//================================================================================================================================
// BLOCK TEST TASKs DETACH
//================================================================================================================================

void activeBlockTest_Detach1()
{
    std::cout <<"Test detach 2 with TasksDispach"<<std::endl;
    auto MyAlgoDetach1 = []{ std::cout <<"I live 1!"<<std::endl; sleep(1);  std::cout <<"YES 1!"<<std::endl;  return 123;};
    auto MyAlgoDetach2 = []{ std::cout <<"I live 2!"<<std::endl; sleep(10); std::cout <<"YES 2!"<<std::endl;  return 123;};
    auto MyAlgoDetach3 = []{ std::cout <<"I live 3!"<<std::endl; sleep(2);  std::cout <<"YES 3!"<<std::endl;  return 123;};

    TasksDispach Fg1; 
    auto t1=Fg1.sub_detach_future_alpha(MyAlgoDetach1);

    TasksDispach Fg2; 
    auto t2=Fg2.sub_detach_future_beta(MyAlgoDetach2);

    TasksDispach Fg3; 
    std::future<int> myfutureFg3=Fg3.sub_detach_future_gamma<int, decltype(MyAlgoDetach3)>(MyAlgoDetach3);
    t1.get(); t2.get(); myfutureFg3.get();
}

void activeBlockTest_Detach2()
{
    int value=123456;
    std::cout<<"Value before="<<value<< std::endl;
    auto Md=[&]() {  
        auto begin = std::chrono::steady_clock::now();
        std::cout << "   I live detach! ..." << std::endl;
        sleep(9);
        value++;
        std::cout << "   YES form detach!." << std::endl;
        auto end = std::chrono::steady_clock::now();
        std::cout << "   ===> dt 2: "<< std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()<< " us\n";std::cout<<"\n"; 
    return true;};

    auto fA=[&]() {  
        auto begin = std::chrono::steady_clock::now();
        std::cout << " I live 4!" << std::endl;
        sleep(2);
        std::cout << " YES 4!" << std::endl;
        auto end = std::chrono::steady_clock::now();
        std::cout << " ===> dt 1: "<< std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()<< " us\n";std::cout<<"\n"; 
        value++;
    return true;};

    auto begin3 = std::chrono::steady_clock::now();
    TasksDispach Fg; 
    Fg.sub_detach_specx_beta(fA,1,Md,1);
    auto end3 = std::chrono::steady_clock::now();
    std::cout << "===> dt 3: "<< std::chrono::duration_cast<std::chrono::microseconds>(end3 - begin3).count()<< " us\n";std::cout<<"\n"; 

    std::cout<<"Value after="<<value<< std::endl;
}


//================================================================================================================================
// BLOCK TEST VECTOR
//================================================================================================================================

void activeBlockTest_Vector(int time_sleep)
{
    int nbThreads = 96;
    auto MyAlgo000=[time_sleep](const int i,double& s) {  
            double sum=0.0; 
            for(int j=0;j<100;j++)
            { 
                sum+=double(j);
            }
            usleep(time_sleep);
            s=sum+i;      
        return true;
    };

    TasksDispach Fg; 
    std::vector<double> valuesVec;

    std::cout<<"With multithread level 2"<< "\n";
    Fg.setFileName("Results"); 
    Fg.init(1,nbThreads,true);  Fg.qInfo=false; Fg.qViewChrono=false; Fg.qSave=false;
    valuesVec=Fg.sub_run_multithread_beta(MyAlgo000);
    std::cout << "Vec R= "; for (int k=0; k<nbThreads; k++) { Color(k+1); std::cout << valuesVec[k] << " ";  } 
    std::cout << "\n"; 
    Color(7);
    std::cout << "\n"; 

    std::cout<<"With std:async level 2"<< "\n";
    Fg.setFileName("With std:async level 2 "); 
    Fg.init(1,nbThreads,true);  Fg.qInfo=false; Fg.qViewChrono=false; Fg.qSave=true;
    valuesVec=Fg.sub_run_async_beta(MyAlgo000);
    std::cout << "Vec R= "; for (int k=0; k<nbThreads; k++) { Color(k+1); std::cout << valuesVec[k] << " ";  } 
    std::cout << "\n"; 
    Color(7);
    std::cout << "\n"; 

    std::cout<<"With specx level 2"<< "\n";
    Fg.init(2,nbThreads,true);  Fg.qInfo=false; Fg.qViewChrono=false; Fg.qSave=true;
    valuesVec=Fg.sub_run_specx(MyAlgo000);
    std::cout << "Vec R= "; for (int k=0; k<nbThreads; k++) { Color(k+1); std::cout << valuesVec[k] << " ";  } 
    std::cout << "\n"; 
    Color(7);
    std::cout << "\n"; 
}


//================================================================================================================================
// BLOCK TEST AFFINITY
//================================================================================================================================

void activeBlockTest_runtask_Affinity()
{   
    int val1=0; 
    //BEGIN::Definition of CPUs used
    std::cout << "List Num CPU="<<"\n";   
    const std::vector<int> NumCPU={1,2,55,80};
    for (int x : NumCPU) { std::cout << x << " "; }
    std::cout << "\n"<< "\n"; 
    //END::Definition of CPUs used

    //BEGIN::Lambda function part: the module that will be executed...
    auto FC1=[&]() {  
        std::cout<<"Hello form CPU="<< sched_getcpu()<<std::endl;
        val1++;
        sleep(1);
        return true;
    };
    //END::Lambda function part

    //BEGIN::Run multithread with TaskDispach
    TasksDispach Fg; 
    Fg.RunTaskInNumCPUs(NumCPU,FC1);
    std::cout << "val1="<<val1<< std::endl;
    //END::Run multithread with TaskDispach
}


//================================================================================================================================
// BLOCK TEST TASKsDISPACH
//================================================================================================================================

void activeBlockTest_runtask_LoopAsync() 
{
    std::cout <<"[TestRunTaskLoop with Async]"<<"\n";
    int val1=0; int val0=0;
    std::cout <<"val1="<<val1<< "\n";

    //BEGIN::Lambda function part: the module that will be executed...
    auto FC1=[&](const int &v,double &w) {  
        std::cout <<"--------------------------------------"<< "\n";
        std::cout <<"Async: I am happy to see you !!!"<< "\n";
        std::cout <<"Val Input="<<v<< "\n";
        std::cout <<"Val Output="<<w<< "\n";
        std::cout <<"--------------------------------------"<< "\n";
        w=w+v;
        usleep(1000);
        val1=val1+99999999;
        return true;
    };
    //END::Lambda function part

    const int valInput1=3;
    double valOutput1=1.5;

    //BEGIN::Run multithread with TaskDispachCompex
    TasksDispachComplex Test1; 
    Test1.numTypeTh=1; //<<< std::Async=1
    Test1.qSave=true;
    Test1.setNbThread(2);
    Test1.run( 
        _parameters=Frontend::parameters(valInput1,valOutput1),
        _task=FC1);

    std::cout <<"val1="<<val1<< " valOutput1="<<valOutput1<< "\n";
    //END::Run multithread with TaskDispachCompex
}


void activeBlockTest_runtask_LoopSpecx()
{
    std::cout <<"[TestRunTaskLoop with Specx]"<<"\n";
    int val1=0; 
    std::cout <<"val1="<<val1<< "\n";
    int val0=0;
 
    //BEGIN::Lambda function part: the module that will be executed...
    auto FC1=[&](const int &v,double &w) {  
        std::cout <<"--------------------------------------"<< "\n";
        std::cout <<"Specx: I am happy to see you !!!"<< "\n";
        std::cout <<"Val Input="<<v<< "\n";
        std::cout <<"Val Output="<<w<< "\n";
        std::cout <<"--------------------------------------"<< "\n";
        w=w+v;
        usleep(1000);
        val1=val1+99999999;
        return true;
    };
    //END::Lambda function part

    const int valInput1=3;
    double valOutput1=1.5;

    //BEGIN::Run multithread with TaskDispachCompex
    TasksDispachComplex Test1; 
    Test1.numTypeTh=2; //<<< Specx=2
    Test1.qSave=true; 
    Test1.setNbThread(2);
    Test1.run(
        _parameters=Frontend::parameters(valInput1,valOutput1),
        _task=FC1);

    std::cout <<"val1="<<val1<< " valOutput1="<<valOutput1<< "\n";
    //END::Run multithread with TaskDispachCompex
}

//================================================================================================================================



int main(int argc, const char** argv) {

  bool qPlayNext=true;
  qPlayNext=false;


  std::cout << std::endl;
  testScanAllThreadMethods();
  std::cout << std::endl;
 
  




// BEGIN::TEST CALCUL INTEGRAL
qPlayNext=true;
qPlayNext=false;
if (qPlayNext) {
    std::cout << std::endl;
    std::cout << "<<< Test calcul intergral  >>>" << std::endl;
    activeBlockTest_Integral001();
    std::cout << std::endl;
    activeBlockTest_Integral002(6);
    std::cout << std::endl;
}
// END::TEST CALCUL INTEGRAL


// BEGIN::TEST ADD VECTORS
qPlayNext=true;
qPlayNext=false;
if (qPlayNext) {
    std::cout << std::endl;
    std::cout << "<<< Test calcul sum vector  >>>" << std::endl;
    activeBlockTest_Add_Vectors();
    std::cout << std::endl;
}
// END::TEST ADD VECTORS

// BEGIN::TEST MATRIX PRODUCT
qPlayNext=true;
qPlayNext=false;
if (qPlayNext) 
{
    std::cout << std::endl;
    std::cout << "<<< Test calcul matrix product  >>>" << std::endl;
    activeBlockTest_Matrix_Products();
    std::cout << std::endl;
}
// END::TEST MATRIX PRODUCT


 // BEGIN::TEST DETACH
qPlayNext=true;
qPlayNext=false;
if (qPlayNext) {
    std::cout << std::endl;
    std::cout << "<<< TEST Detach >>>" << std::endl;
    activeBlockTest_Detach1();
    std::cout << std::endl;

    std::cout << std::endl;
    std::cout << "<<< TEST Detach with Specx  >>>" << std::endl;
    activeBlockTest_Detach2();
    std::cout << std::endl;
}
// END::TEST DETACH

 
// BEGIN::TEST MULTITHREAD AFFINITY
qPlayNext=true;
qPlayNext=false;
if (qPlayNext) {
    std::cout << std::endl;
    std::cout << "<<< TEST Multithread Affinity  >>>" << std::endl;
    activeBlockTest_runtask_Affinity();
    std::cout << std::endl;
}
// END::TEST MULTITHREAD AFFINITY


// BEGIN::TEST VECTOR with all configuration
qPlayNext=true;
qPlayNext=false;
if (qPlayNext) {
    std::cout << std::endl;
    std::cout << "<<< TEST Vector  >>>" << std::endl;
    activeBlockTest_Vector(100000);
    std::cout << std::endl;
}
// END::TEST VECTOR with all configuration


// BEGIN::TEST TasksDispachComplex LoopAsync
qPlayNext=true;
//qPlayNext=false;
if (qPlayNext) {
    std::cout << std::endl;
    std::cout << "<<< TEST TasksDispachComplex LoopAsync part  >>>" << std::endl;
    activeBlockTest_runtask_LoopAsync();
    std::cout << std::endl;
}
// END::TEST TasksDispachComplex LoopAsync


// BEGIN::TEST TasksDispachComplex LoopSpecx
qPlayNext=true;
//qPlayNext=false;
if (qPlayNext) {
    std::cout << std::endl;
    std::cout << "<<< TEST TasksDispachComplex LoopSpecx part  >>>" << std::endl;
    activeBlockTest_runtask_LoopSpecx();
    std::cout << std::endl;
}
// END::TEST TasksDispachComplex LoopSpecx


  std::cout << "<<< The End : Well Done >>>" << std::endl << std::endl;
  Color(7);
  return 0;
}
