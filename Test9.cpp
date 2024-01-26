
#include <thread>
#include <vector>
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


#define clrscr() printf("\033[H\033[2J")
#define color(param) printf("\033[%sm",param)

/* 
    0  réinitialisation         1  haute intensité (des caractères)
    5  clignotement             7  video inversé
    30, 31, 32, 33, 34, 35, 36, 37 couleur des caractères
    40, 41, 42, 43, 44, 45, 46, 47 couleur du fond
    noir, rouge, vert, jaune, bleu, magenta, cyan et blanc 
    0,      1,    2,     3,     4,     5,      6,      7
    color("40;37")               
*/


void Color(int number)
{
    if (number>7) { number=number % 8; }
    number+=30;
    printf("\033[%im",number);
}

void ColorBackground(int number)
{
    if (number>7) { number=number % 8; }
    number+=40;
    printf("\033[%im",number);
}


/*=====================================================================================================*/

//ADD class Artifac 




class TasksDispach
{
    private:
        int nbThTotal;   
        std::vector<int> indice;
        void initIndice();

    public:
        int numTypeTh;
        int nbTh;
        bool QSave;
        bool QSlipt;
        bool QDeferred;
        std::string FileName;

        auto begin(); 
        auto end(); 
        auto size(); 

        int getNbMaxThread();
        void init(int numType,int nbThread,bool QsaveInfo);
        void setFileName(std::string s);
        template<class Function>
            Function run(Function myFunc);
                template<class Function>
                    Function sub_run_async(Function myFunc);
                template<class Function>
                    Function sub_run_specx_W(Function myFunc);
                template<class Function>
                    Function sub_run_specx_R(Function myFunc);

        template<class Function,class ArgR,class ArgW>
            Function sub_run_specx_RW(Function myFunc,ArgR myArgR,ArgW myArgW);

        template<class InputIterator, class Function>
            Function for_each(InputIterator first, InputIterator last,Function myFunc);

        TasksDispach(void);
        ~TasksDispach(void);
};


TasksDispach::TasksDispach() { 
    nbThTotal=std::thread::hardware_concurrency();
    numTypeTh=2; 
    nbTh=1;
    QSave=false;
    FileName="TestDispach";
    QSlipt=false;
    QDeferred=false;
    initIndice();
}

TasksDispach::~ TasksDispach(void) { 
}


auto TasksDispach::begin()
{
    return(indice.begin());
}

auto TasksDispach::end()
{
    return(indice.end());
}

auto TasksDispach::size()
{
    return(indice.size());
}

void TasksDispach::initIndice()
{
    indice.clear();
    //std::cout<<"nbTh="<<nbTh<<" Indice size="<<indice.size()<<"\n";
    //indice(nbTh,0);
    for (int i = 1; i <= nbTh; ++i)  { indice.push_back(i); }
    //for (int x : indice) { std::cout << x << " "; }
    //std::cout<<"\n";
    //std::cout<<"nbTh="<<nbTh<<" Indice size="<<indice.size()<<"\n";
}

void TasksDispach::init(int numType,int nbThread,bool QsaveInfo)
{
    numTypeTh=numType; nbTh=nbThread; QSave=QsaveInfo;
    initIndice();
}


void TasksDispach::setFileName(std::string s)
{
    FileName=s;
}

int TasksDispach::getNbMaxThread()
{
    nbThTotal=std::thread::hardware_concurrency();
    return(nbThTotal);
}


template<class InputIterator, class Function>
Function TasksDispach::for_each(InputIterator first, InputIterator last,Function myFunc)
{
    bool QInfo=true;
        if (numTypeTh==1) {
            std::vector< std::future< bool > > futures;
            for ( ; first!=last; ++first )
            { 
                auto const& idk = *first;
                if (QInfo) { std::cout<<"Call num Thread futures="<<idk<<"\n"; }
                if (QDeferred) { futures.emplace_back(std::async(std::launch::deferred,myFunc,idk)); }
                else { futures.emplace_back(std::async(std::launch::async,myFunc,idk)); }
            }
            for( auto& r : futures){ auto a =  r.get(); }
        }

        if (numTypeTh==2) {
            SpRuntime runtime(nbTh);  
            nbTh= runtime.getNbThreads();
            for ( ; first!=last; ++first )
            { 
                auto const& idk = *first;
                if (QInfo) { std::cout<<"Call num Thread Read Specx="<<idk<<"\n"; }
                runtime.task(SpRead(idk),myFunc).setTaskName("Op("+std::to_string(idk)+")");
                usleep(1);
                std::atomic_int counter(0);
            }
            runtime.waitAllTasks();
            runtime.stopAllThreads();
            if (QSave)
            {
                runtime.generateDot(FileName+".dot", true);
                runtime.generateTrace(FileName+".svg");   
            }
        }
        if (QInfo) { std::cout<<"\n"; }
    return myFunc;
}


template<class Function>
Function TasksDispach::sub_run_async(Function myFunc)
{
   bool QInfo=true; 
        std::vector< std::future< bool > > futures;
        for(int k= 0; k < nbTh; ++k){ 
            auto const& idk = k;
            if (QInfo) { std::cout<<"Call num Thread futures="<<k<<"\n"; }
            if (QDeferred) { futures.emplace_back(std::async(std::launch::deferred,myFunc,idk)); }
            else { futures.emplace_back(std::async(std::launch::async,myFunc,idk)); }
        }
        for( auto& r : futures){ auto a =  r.get(); }
        if (QInfo) { std::cout<<"\n"; }
    return myFunc;
}

template<class Function>
Function TasksDispach::sub_run_specx_W(Function myFunc)
{
   bool QInfo=true; 
        SpRuntime runtime(nbTh);  
        nbTh= runtime.getNbThreads();
        int iValue=0;
        std::vector<int> valuesVec(nbTh,0); //trick to launch everything at once

        for(int k= 0; k < nbTh; ++k)
        { 
            auto const& idk = k;
            if (QInfo) { std::cout<<"Call num Thread Write Specx="<<idk<<"\n"; }
                if (QSlipt) {
                    runtime.task(SpWrite(iValue),myFunc).setTaskName("Op("+std::to_string(idk)+")");
                }
                else
                {
                    runtime.task(SpWrite(valuesVec.at(idk)),myFunc).setTaskName("Op("+std::to_string(idk)+")");
                }
            usleep(1);
            std::atomic_int counter(0);
        }
        runtime.waitAllTasks();
        runtime.stopAllThreads();
        if (QSave)
        {
            runtime.generateDot(FileName+".dot", true);
            runtime.generateTrace(FileName+".svg");   
        }
        if (QInfo) { std::cout<<"\n"; }
    return myFunc;
}


template<class Function>
Function TasksDispach::sub_run_specx_R(Function myFunc)
{
   bool QInfo=true; 
        SpRuntime runtime(nbTh);  
        nbTh= runtime.getNbThreads();
        int iValue=0;
        for(int k= 0; k < nbTh; ++k)
        { 
            auto const& idk = k;
            if (QInfo) { std::cout<<"Call num Thread Read Specx="<<idk<<"\n"; }
            runtime.task(SpRead(idk),myFunc).setTaskName("Op("+std::to_string(idk)+")");
            usleep(1);
            std::atomic_int counter(0);
        }
        runtime.waitAllTasks();
        runtime.stopAllThreads();
        if (QSave)
        {
            runtime.generateDot(FileName+".dot", true);
            runtime.generateTrace(FileName+".svg");   
        }
        if (QInfo) { std::cout<<"\n"; }
    return myFunc;
}

template<class Function,class ArgR,class ArgW>
Function TasksDispach::sub_run_specx_RW(Function myFunc,ArgR myArgR,ArgW myArgW)
{
   bool QInfo=true; 
        SpRuntime runtime(nbTh);  
        nbTh= runtime.getNbThreads();
        int iValue=0;
        for(int k= 0; k < nbTh; ++k)
        { 
            auto const& idk = k;
            if (QInfo) { std::cout<<"Call num Thread Read Specx="<<idk<<"\n"; }
            runtime.task(SpRead(myArgR),SpWrite(myArgW),myFunc).setTaskName("Op("+std::to_string(idk)+")");
            usleep(1);
            std::atomic_int counter(0);
        }
        runtime.waitAllTasks();
        runtime.stopAllThreads();
        if (QSave)
        {
            runtime.generateDot(FileName+".dot", true);
            runtime.generateTrace(FileName+".svg");   
        }
        if (QInfo) { std::cout<<"\n"; }
    return myFunc;
}


template<class Function>
Function TasksDispach::run(Function myFunc)
{
   if (numTypeTh==1) return sub_run_async(myFunc);
   if (numTypeTh==2) return sub_run_specx_R(myFunc);
   if (numTypeTh==3) return sub_run_specx_W(myFunc);
}


/*=====================================================================================================*/

template<class Function, class... Args>
void async_wrapper(Function&& f, Args&&... args, std::future<void>& future,
                   std::future<void>&& is_valid, std::promise<void>&& is_moved) {
    is_valid.wait(); // Wait until the return value of std::async is written to "future"
    auto our_future = std::move(future); // Move "future" to a local variable
    is_moved.set_value(); // Only now we can leave void_async in the main thread

    // This is also used by std::async so that member function pointers work transparently
    auto functor = std::bind(f, std::forward<Args>(args)...);
    functor();
}

template<class Function, class... Args> // This is what you call instead of std::async
void void_async(Function&& f, Args&&... args) {
    std::future<void> future; // This is for std::async return value
    // This is for our synchronization of moving "future" between threads
    std::promise<void> valid;
    std::promise<void> is_moved;
    auto valid_future = valid.get_future();
    auto moved_future = is_moved.get_future();

    // Here we pass "future" as a reference, so that async_wrapper
    // can later work with std::async's return value
    future = std::async(
        async_wrapper<Function, Args...>,
        std::forward<Function>(f), std::forward<Args>(args)...,
        std::ref(future), std::move(valid_future), std::move(is_moved)
    );
    valid.set_value(); // Unblock async_wrapper waiting for "future" to become valid
    moved_future.wait(); // Wait for "future" to actually be moved
}


int long_running_task(int target, const std::atomic_bool& cancelled)
{
    // simulate a long running task for target*100ms, 
    // the task should check for cancelled often enough!
    while(target-- && !cancelled)
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    // return results to the future or raise an error 
    // in case of cancellation
    return cancelled ? 1 : 0;
}



/*=====================================================================================================*/

template <typename result_type, typename function_type>
std::future<result_type> startdetachedfuture(function_type func) {
    std::promise<result_type> pro;
    std::future<result_type> fut = pro.get_future();
    std::thread([func](std::promise<result_type> p){p.set_value(func());},
                std::move(pro)).detach();

    return fut;
}


template<typename F>
auto async_deferred_alpha(F&& func) -> std::future<decltype(func())>
{
    auto task   = std::packaged_task<decltype(func())()>(std::forward<F>(func));
    auto future = task.get_future();
    std::thread(std::move(task)).detach();
    return std::move(future);
}

template<typename F>
auto async_deferred_beta(F&& func) -> std::future<decltype(func())>
{
    using result_type = decltype(func());
    auto promise = std::promise<result_type>();
    auto future  = promise.get_future();
    std::thread(std::bind([=](std::promise<result_type>& promise)
    {
        try
        {
            promise.set_value(func()); 
        }
        catch(...)
        {
            promise.set_exception(std::current_exception());
        }
    }, std::move(promise))).detach();
    return future;
}

/*=====================================================================================================*/



template<class Function>
Function TestSpecxWithTwoParam(Function myFunc)
{
    SpRuntime runtime(6);  
    int nbThread= runtime.getNbThreads();
    int iValue=0;
    for(int k= 0; k < nbThread; ++k)
    { 
        auto const& idk = k;
        runtime.task(SpRead(idk),SpWrite(iValue),myFunc).setTaskName("Op("+std::to_string(idk)+")");
        usleep(1);
        std::atomic_int counter(0);
    }
    runtime.waitAllTasks();
    runtime.stopAllThreads();
    runtime.generateDot("TestSpecxWithTwoParam.dot", true);
    runtime.generateTrace("TestSpecxWithTwoParam.svg");  
    std::cout<<"\n";
    return myFunc;
}

template<class Function,class ArgR,class ArgW>
Function TestSpecxWith3Param(Function myFunc,ArgR myArgR,ArgW myArgW)
{
   bool QInfo=true; 
    SpRuntime runtime(6);  
    int nbTh= runtime.getNbThreads();
        int iValue=0;
        for(int k= 0; k < nbTh; ++k)
        { 
            auto const& idk = k;
            if (QInfo) { std::cout<<"Call num Thread Read Specx="<<idk<<"\n"; }
            runtime.task(SpRead(myArgR),SpWrite(myArgW),myFunc).setTaskName("Op("+std::to_string(idk)+")");
            usleep(1);
            std::atomic_int counter(0);
        }
        runtime.waitAllTasks();
        runtime.stopAllThreads();
        runtime.generateDot("TestSpecxWith3Param.dot", true);
        runtime.generateTrace("TestSpecxWith3Param.svg");   
        std::cout<<"\n"; 
    return myFunc;
}


void activeBlock000()
{
//BEGIN:Test with Specx
        int wValueOut=0;
        std::cout<<"TEST WITH SPECX"<<std::endl;
        auto MyAlgo9=[&](const int& k,int& v) {  
            std::cout<<"wValue k="<<k<< std::endl;
            wValueOut++; 
            usleep(1000);
        return true;};

        wValueOut=0;
        std::cout << std::endl;
        std::cout<<"Test async inc k"<< std::endl;
        std::cout<<"wValueOut="<<wValueOut<< std::endl;
        TestSpecxWithTwoParam(MyAlgo9);
        std::cout<<"wValueOut="<<wValueOut<< std::endl;
        std::cout<<"-------------------------------"<<std::endl;
    //END:Test woth Specx
}


/*=====================================================================================================*/
/*=====================================================================================================*/

void activeBlockTest001()
{
    //CALCUL DE PI par les deux méthodes std::async et Specx
    int nbThreads = 9;
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
        return true;
    };


    std::cout<<"PI method Specx"<<"\n";
    valuesVec.clear(); std::cout<<"Clear results size="<<valuesVec.size()<< "\n";
    TasksDispach FgCalculIntegral; 
    FgCalculIntegral.init(2,nbThreads,true); FgCalculIntegral.setFileName("TestDispachIntegral");
    FgCalculIntegral.run(MyAlgo000);
    std::cout << "Vec R= "; 
    for (int k=0; k<valuesVec.size(); k++) { Color(k);std::cout << valuesVec[k] << " "; }
    Color(7);
    std::cout << "\n"; 
    integralValue=h*std::reduce(valuesVec.begin(),valuesVec.end()); 
    std::cout<<"PI Value= "<<integralValue<<"\n";

    std::cout<<"PI method std::async"<<"\n";
    valuesVec.clear(); std::cout<<"Clear results size="<<valuesVec.size()<< "\n";
    FgCalculIntegral.init(1,nbThreads,true); FgCalculIntegral.setFileName("TestDispachIntegral");
    FgCalculIntegral.run(MyAlgo000);
    std::cout << "Vec R= "; 
    for (int k=0; k<valuesVec.size(); k++) { Color(k);std::cout << valuesVec[k] << " "; }
    Color(7);

    std::cout << "\n"; 
    integralValue=h*std::reduce(valuesVec.begin(),valuesVec.end()); 
    std::cout<<"PI Value= "<<integralValue<<"\n";
}

void activeBlockTest002()
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
    FgCalcul.init(1,nbThreads,true); FgCalcul.setFileName("TestDispachSum");
    FgCalcul.run(MyAlgo000);
    std::cout << "Vec A= "; 
    for (auto it = VecA.begin(); it != VecA.end(); it++) { std::cout << *it << " "; }
    std::cout << "\n"; 
    std::cout << "Vec B= "; 
    for (auto it = VecB.begin(); it != VecB.end(); it++) { std::cout << *it << " "; }
    std::cout << "\n"; 
    std::cout << "Vec R= "; 
    for (int k=0; k<VecR.size(); k++) { Color(k/sizeBlock);std::cout << VecR[k] << " "; }
    Color(7);

    std::cout << "\n";
    std::cout << "\n"; 
    std::cout<<"Calcul with Specx"<<"\n";
    VecR.clear();
    FgCalcul.init(2,nbThreads,true); FgCalcul.setFileName("TestDispachSum");
    FgCalcul.run(MyAlgo000);
    std::cout << "Vec A= "; 
    for (auto it = VecA.begin(); it != VecA.end(); it++) { std::cout << *it << " "; }
    std::cout << "\n"; 
    std::cout << "Vec B= "; 
    for (auto it = VecB.begin(); it != VecB.end(); it++) { std::cout << *it << " "; }
    std::cout << "\n"; 
    std::cout << "Vec R= "; 
    for (int k=0; k<VecR.size(); k++) { Color(k/sizeBlock);std::cout << VecR[k] << " "; }
    Color(7);
    std::cout << "\n"<< "\n"; 
}


/*=====================================================================================================*/
/*=====================================================================================================*/


void activeBlock001()
{
    int nbThreads=6;
    int wValueOut=0;

    int nbThreadsTotal = std::thread::hardware_concurrency();
    std::cout<<"nbThreadsTotal="<<nbThreadsTotal<< std::endl;
    int numCPU = sysconf(_SC_NPROCESSORS_ONLN);
    std::cout<<"numCPU="<<numCPU<< std::endl;


    auto MyAlgo000=[&](const int& k) {  
        //std::cout<<"wValue k="<<k<< std::endl;
        wValueOut++; 
        usleep(1000);
    return true;};

    std::cout<<"[WITH STD::ASYNC]"<< std::endl;
    std::cout<<"wValueOut="<<wValueOut<< std::endl;
    auto start_time= std::chrono::steady_clock::now();
    TasksDispach Fg1; 
    Fg1.init(1,12,true); Fg1.setFileName("TestDispachAsync"); //Fg1.QDeferred=true;
    Fg1.sub_run_async(MyAlgo000);
    auto stop_time= std::chrono::steady_clock::now();
    auto run_time=std::chrono::duration_cast<std::chrono::microseconds> (stop_time-start_time);
    std::cout<<"DeltaTime="<<run_time.count()<< std::endl;
    std::cout<<"wValueOut="<<wValueOut<< std::endl<<std::endl;

    std::cout<<"[WITH STD::SPECX]"<< std::endl;
    std::cout<<"wValueOut="<<wValueOut<< std::endl;
    start_time= std::chrono::steady_clock::now();
    TasksDispach Fg2; 
    Fg2.init(2,12,true); Fg2.setFileName("TestDispachSpecxR");
    Fg2.sub_run_specx_R(MyAlgo000);
    stop_time= std::chrono::steady_clock::now();
    run_time=std::chrono::duration_cast<std::chrono::microseconds> (stop_time-start_time);
    std::cout<<"DeltaTime="<<run_time.count()<< std::endl;
    std::cout<<"wValueOut="<<wValueOut<< std::endl<<std::endl;

    std::cout<<"[WITH STD::ASYNC call run]"<< std::endl;
    std::cout<<"wValueOut="<<wValueOut<< std::endl;
    start_time= std::chrono::steady_clock::now();
    TasksDispach Fg3; 
    Fg3.init(1,12,true); Fg3.setFileName("TestDispach");
    Fg3.run(MyAlgo000);
    stop_time= std::chrono::steady_clock::now();
    run_time=std::chrono::duration_cast<std::chrono::microseconds> (stop_time-start_time);
    std::cout<<"DeltaTime="<<run_time.count()<< std::endl;
    std::cout<<"wValueOut="<<wValueOut<< std::endl<<std::endl;

    std::cout<<"[WITH STD::SPECX call run]"<< std::endl;
    std::cout<<"wValueOut="<<wValueOut<< std::endl;
    start_time= std::chrono::steady_clock::now();
    TasksDispach Fg4; 
    Fg4.init(2,12,true); Fg4.setFileName("TestDispach");
    Fg4.run(MyAlgo000);
    stop_time= std::chrono::steady_clock::now();
    run_time=std::chrono::duration_cast<std::chrono::microseconds> (stop_time-start_time);
    std::cout<<"DeltaTime="<<run_time.count()<< std::endl;
    std::cout<<"wValueOut="<<wValueOut<< std::endl<<std::endl;

  
    auto MyAlgo005=[&](int& v) {  
        //std::cout<<"wValue k="<<k<< std::endl;
        wValueOut++; 
        usleep(1000);
    return true;};

    std::cout<<"wValueOut="<<wValueOut<< std::endl;
    start_time= std::chrono::steady_clock::now();
    TasksDispach Fg5; 
    Fg5.init(3,12,true); Fg5.setFileName("TestDispachSpecxW"); 
    Fg5.QSlipt=true;
    Fg5.sub_run_specx_W(MyAlgo005);
    //Fg5.run(MyAlgo005);
    stop_time= std::chrono::steady_clock::now();
    run_time=std::chrono::duration_cast<std::chrono::microseconds> (stop_time-start_time);
    std::cout<<"DeltaTime="<<run_time.count()<< std::endl;
    std::cout<<"wValueOut="<<wValueOut<< std::endl<<std::endl;

    std::cout <<"END:MyEachThread1\n\n";
}


void activeBlock002()
{
    int nbThreads=6;
    int wValueOut=0;

    auto MyAlgo000=[&](const int& k) {  
        //std::cout<<"wValue k="<<k<< std::endl;
        wValueOut++; 
        usleep(1000);
    return true;};

    /*
    std::cout << std::endl;
    std::cout <<"BEGIN:ForEach 1\n";
    std::vector<int> valuesVec(nbThreads,0); for (int k=0; k<nbThreads; ++k) { valuesVec[k]=k+1; }
    TasksDispach Fg1; 
    Fg1.init(2,nbThreads,true); Fg1.setFileName("TestForEach");
    Fg1.for_each(valuesVec.begin()+2,valuesVec.end(),MyAlgo000);
    std::cout <<"\n";
    */

    std::cout << std::endl;
    std::cout <<"BEGIN:ForEach 2\n";
    TasksDispach Fg2; 
    Fg2.init(2,3,true);
    //Fg2.init(2,5,true);
    Fg2.for_each(Fg2.begin(),Fg2.end(),MyAlgo000);
    std::cout <<"\n";
    std::cout <<"END:ForEach\n\n";
}


void activeBlock003()
{
    std::cout <<"Test detach 1"<<std::endl;
    auto MyAlgoDetach = []{ std::cout <<"I live!"<<std::endl; sleep(2); std::cout <<"YES!"<<std::endl;  return 123;};
    std::future<int> myfuture = startdetachedfuture<int, decltype(MyAlgoDetach)>(MyAlgoDetach);
    std::cout <<"Hello"<<std::endl;  sleep(10);  std::cout <<"Hello2"<<std::endl;
}


void activeBlock004()
{
    std::cout <<"Test detach 2"<<std::endl;
    // future from a packaged_task
    std::packaged_task<int()> task([]{ return 7; }); // wrap the function

    std::future<int> f1 = task.get_future(); 

    std::thread t(std::move(task)); // launch on a thread
 
    // future from an async()
    std::future<int> f2 = std::async(std::launch::async, []{ return 8; });
 
    // future from a promise
    std::promise<int> p;
    std::future<int> f3 = p.get_future();
    std::thread([&p]{ p.set_value_at_thread_exit(9); }).detach();
 
    std::cout << "Waiting..." << std::flush;
    f1.wait();
    f2.wait();
    f3.wait();
    std::cout << "Done!\nResults are: "
              << f1.get() << ' ' << f2.get() << ' ' << f3.get() << '\n';
    t.join();

}

void activeBlock005()
{
    std::cout <<"Test detach 3"<<std::endl;
    auto MyAlgoDetach2 = []{ std::cout <<"I live 2!"<<std::endl; sleep(10); std::cout <<"YES 2!"<<std::endl;  return 123;};
    auto MyAlgoDetach3 = []{ std::cout <<"I live 3!"<<std::endl; sleep(1); std::cout <<"YES 3!"<<std::endl;  return 123;};
    auto t1=async_deferred_alpha(MyAlgoDetach2); 
    auto t2=async_deferred_beta(MyAlgoDetach3);
    t1.get(); t2.get();
}


int main(int argc, const char** argv) {


  //std::cout << std::endl;
  //std::cout << "<<< Block000 >>>" << std::endl;
  //activeBlock000();
  //std::cout << std::endl;

  std::cout << std::endl;
  std::cout << "<<< Block001 >>>" << std::endl;
  activeBlock001();
  std::cout << std::endl;

  std::cout << std::endl;
  std::cout << "<<< Block002 >>>" << std::endl;
  activeBlock002();
  std::cout << std::endl;

  // BEGIN::TEST BENCHMARKS
  std::cout << std::endl;
  std::cout << "<<< ====================================== >>>" << std::endl;
  std::cout << "<<< Test calcul intergral  >>>" << std::endl;
  activeBlockTest001();
  std::cout << std::endl;

  std::cout << std::endl;
  std::cout << "<<< ====================================== >>>" << std::endl;
  std::cout << "<<< Test calcul sum vector  >>>" << std::endl;
  activeBlockTest002();
  std::cout << std::endl;
  // BEGIN::END BENCHMARKS

  std::cout << std::endl;
  std::cout << "<<< Block003: Detached Future >>>" << std::endl;
  activeBlock003();
  std::cout << std::endl;

  std::cout << std::endl;
  std::cout << "<<< Block004: Detached Future 2 >>>" << std::endl;
  activeBlock004();
  std::cout << std::endl;

  std::cout << std::endl;
  std::cout << "<<< Block005: Deferred >>>" << std::endl;
  activeBlock005();
  std::cout << std::endl;






  std::cout << "<<< The End >>>" << std::endl << std::endl;
  Color(7);
  return 0;
}
