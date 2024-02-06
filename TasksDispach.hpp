
#include <thread>
#include <vector>
#include <array>
#include <typeinfo>
#include <iostream>
#include <mutex>
#include <sched.h>
#include <pthread.h>

#include<algorithm> 
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


class TasksDispach
{
    private:
        int nbThTotal;   
        std::vector<int> indice;
        void initIndice();

    public:
        int numTypeTh;
        int nbTh;
        bool qSave;
        bool qSlipt;
        bool qDeferred;
        bool qViewChrono;
        bool qInfo;
        std::string FileName;

        auto begin(); 
        auto end(); 
        auto size(); 

        int getNbMaxThread();
        void init(int numType,int nbThread,bool qsaveInfo);
        void setFileName(std::string s);
        template<class Function>
            Function run(Function myFunc);
                template<class Function>
                    Function sub_run_multithread(Function myFunc);
                template<class Function>
                    Function sub_run_async(Function myFunc);
                template<class Function>
                    Function sub_run_specx_W(Function myFunc);
                template<class Function>
                    Function sub_run_specx_R(Function myFunc);

                template<class Function> 
                    std::vector<double> sub_run_specx(Function myFunc);
                template<class Function>
                    std::vector<double> sub_run_async_beta(Function myFunc);

        //BEGIN::Detach part
        template<typename FctDetach>
                    auto sub_detach_future_alpha(FctDetach&& func) -> std::future<decltype(func())>;
        template<typename FctDetach>
                    auto sub_detach_future_beta(FctDetach&& func) -> std::future<decltype(func())>;
        template <typename result_type, typename FctDetach>
                    std::future<result_type> sub_detach_future_gamma(FctDetach func);
        //END::Detach part

        template<class ArgR,class ArgW,class Function>
            Function sub_run_specx_RW(ArgR myArgR,ArgW myArgW,Function myFunc);

        template<class InputIterator, class Function>
            Function for_each(InputIterator first, InputIterator last,Function myFunc);

        TasksDispach(void);
        ~TasksDispach(void);
};


TasksDispach::TasksDispach() { 
    nbThTotal=std::thread::hardware_concurrency();
    numTypeTh=2; 
    nbTh=1;
    qSave=false;
    FileName="TestDispach";
    qSlipt=false;
    qDeferred=false;
    qViewChrono=true;
    qViewChrono=false;
    qInfo=false;  
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
    for (int i = 1; i <= nbTh; ++i)  { indice.push_back(i); }
}

void TasksDispach::init(int numType,int nbThread,bool qsaveInfo)
{
    numTypeTh=numType; nbTh=nbThread; qSave=qsaveInfo; qInfo=false;  
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


template<typename FctDetach>
auto TasksDispach::sub_detach_future_alpha(FctDetach&& func) -> std::future<decltype(func())>
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

template<typename FctDetach>
auto TasksDispach::sub_detach_future_beta(FctDetach&& func) -> std::future<decltype(func())>
{
    auto task   = std::packaged_task<decltype(func())()>(std::forward<FctDetach>(func));
    auto future = task.get_future();
    std::thread(std::move(task)).detach();
    return std::move(future);
}


template <typename result_type, typename FctDetach>
    std::future<result_type> TasksDispach::sub_detach_future_gamma(FctDetach func) {
    std::promise<result_type> pro;
    std::future<result_type> fut = pro.get_future();
    std::thread([func](std::promise<result_type> p){p.set_value(func());},std::move(pro)).detach();
    return fut;
}


template<class InputIterator, class Function>
Function TasksDispach::for_each(InputIterator first, InputIterator last,Function myFunc)
{
        if (numTypeTh==1) {
            std::vector< std::future< bool > > futures;
            for ( ; first!=last; ++first )
            { 
                auto const& idk = *first;
                if (qInfo) { std::cout<<"Call num Thread futures="<<idk<<"\n"; }
                if (qDeferred) { futures.emplace_back(std::async(std::launch::deferred,myFunc,idk)); }
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
                if (qInfo) { std::cout<<"Call num Thread Read Specx="<<idk<<"\n"; }
                runtime.task(SpRead(idk),myFunc).setTaskName("Op("+std::to_string(idk)+")");
                usleep(1);
                std::atomic_int counter(0);
            }
            runtime.waitAllTasks();
            runtime.stopAllThreads();
            if (qSave)
            {
                runtime.generateDot(FileName+".dot", true);
                runtime.generateTrace(FileName+".svg");   
            }
        }
        if (qInfo) { std::cout<<"\n"; }
    return myFunc;
}


template<class Function>
Function TasksDispach::sub_run_multithread(Function myFunc)
{
        auto begin = std::chrono::steady_clock::now();
        std::vector<std::thread> mythreads;
        for(int k= 0; k < nbTh; ++k){ 
            auto const& idk = k;
            if (qInfo) { std::cout<<"Call num Multithread ="<<k<<"\n"; }
            std::thread th(myFunc,idk);
            mythreads.push_back(move(th));
        }
        for (std::thread &t : mythreads) {
            t.join();
        }
        auto end = std::chrono::steady_clock::now();
        if (qInfo) { std::cout<<"\n"; }
        if (qViewChrono) {  std::cout << "===> Elapsed microseconds: "<< std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()<< " us\n"; std::cout<<"\n"; }
    return myFunc;
}




template<class Function>
Function TasksDispach::sub_run_async(Function myFunc)
{
        auto begin = std::chrono::steady_clock::now();
        std::vector< std::future< bool > > futures;
        for(int k= 0; k < nbTh; ++k){ 
            auto const& idk = k;
            if (qInfo) { std::cout<<"Call num Thread futures="<<k<<"\n"; }
            if (qDeferred) { futures.emplace_back(std::async(std::launch::deferred,myFunc,idk)); }
            else { futures.emplace_back(std::async(std::launch::async,myFunc,idk)); }
        }
        for( auto& r : futures){ auto a =  r.get(); }
        auto end = std::chrono::steady_clock::now();
        if (qInfo) { std::cout<<"\n"; }
        if (qViewChrono) {  std::cout << "===> Elapsed microseconds: "<< std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()<< " us\n";std::cout<<"\n"; }
    return myFunc;
}

template<class Function>
std::vector<double> TasksDispach::sub_run_async_beta(Function myFunc)
{
        std::vector<double> valuesVec(nbTh,0);
        auto begin = std::chrono::steady_clock::now();
        auto LF=[&](const int& k) {  myFunc(k,valuesVec.at(k)); return true;};
        std::vector< std::future< bool > > futures;
        for(int k= 0; k < nbTh; ++k){ 
            auto const& idk = k;
            if (qInfo) { std::cout<<"Call num Thread futures="<<k<<"\n"; }
            futures.emplace_back(std::async(std::launch::async,LF,idk)); 
        }
        for( auto& r : futures){ auto a =  r.get(); }
        auto end = std::chrono::steady_clock::now();
        if (qInfo) { std::cout<<"\n"; }
        if (qViewChrono) {  std::cout << "===> Elapsed microseconds: "<< std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()<< " us\n";std::cout<<"\n"; }
    return valuesVec;
}

template<class Function>
Function TasksDispach::sub_run_specx_W(Function myFunc)
{
        SpRuntime runtime(nbTh);  
        auto begin = std::chrono::steady_clock::now();
        nbTh= runtime.getNbThreads();
        int iValue=0;
        std::vector<int> valuesVec(nbTh,0); //trick to launch everything at once

        for(int k= 0; k < nbTh; ++k)
        { 
            auto const& idk = k;
            if (qInfo) { std::cout<<"Call num Thread Write Specx="<<idk<<"\n"; }
                if (qSlipt) {
                    runtime.task(SpWrite(iValue),myFunc).setTaskName("Op("+std::to_string(idk)+")");
                }
                else
                {
                    runtime.task(SpWrite(valuesVec.at(idk)),myFunc).setTaskName("Op("+std::to_string(idk)+")");
                }
            //usleep(1);
            //std::atomic_int counter(0);
        }
        runtime.waitAllTasks();
        runtime.stopAllThreads();
        auto end = std::chrono::steady_clock::now();
        if (qSave)
        {
            runtime.generateDot(FileName+".dot", true);
            runtime.generateTrace(FileName+".svg");   
        }
        if (qInfo) { std::cout<<"\n"; }
        if (qViewChrono) {  std::cout << "===> Elapsed microseconds: "<< std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()<< " us\n";std::cout<<"\n"; }
    return myFunc;
}


template<class Function>
std::vector<double> TasksDispach::sub_run_specx(Function myFunc)
{
        std::vector<double> valuesVec(nbTh,0); 
        SpRuntime runtime(nbTh);  
        auto begin = std::chrono::steady_clock::now();
        nbTh= runtime.getNbThreads();
        for(int k= 0; k < nbTh; ++k)
        { 
            auto const& idk = k;
            if (qInfo) { std::cout<<"Call num Thread Write Specx="<<idk<<"\n"; }
            runtime.task(SpRead(idk),SpWrite(valuesVec.at(idk)),myFunc).setTaskName("Op("+std::to_string(idk)+")");
            usleep(0); std::atomic_int counter(0);
        }
        runtime.waitAllTasks();
        runtime.stopAllThreads();
        auto end = std::chrono::steady_clock::now();
        if (qSave)
        {
            runtime.generateDot(FileName+".dot", true);
            runtime.generateTrace(FileName+".svg");   
        }
        if (qInfo) { std::cout<<"\n"; }
        if (qViewChrono) {  std::cout << "===> Elapsed microseconds: "<< std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()<< " us\n";std::cout<<"\n"; }

    return valuesVec;
}



template<class Function>
Function TasksDispach::sub_run_specx_R(Function myFunc)
{
        SpRuntime runtime(nbTh);  
        auto begin = std::chrono::steady_clock::now();
        nbTh= runtime.getNbThreads();
        int iValue=0;
        for(int k= 0; k < nbTh; ++k)
        { 
            auto const& idk = k;
            if (qInfo) { std::cout<<"Call num Thread Read Specx="<<idk<<"\n"; }
            runtime.task(SpRead(idk),myFunc).setTaskName("Op("+std::to_string(idk)+")");
            usleep(0);
            std::atomic_int counter(0);
        }
        runtime.waitAllTasks();
        runtime.stopAllThreads();
        auto end = std::chrono::steady_clock::now();
        if (qSave)
        {
            runtime.generateDot(FileName+".dot", true);
            runtime.generateTrace(FileName+".svg");   
        }
        
        if (qInfo) { std::cout<<"\n"; }
        if (qViewChrono) {  std::cout << "===> Elapsed microseconds: "<< std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()<< " us\n"; std::cout<<"\n"; }
    return myFunc;
}




template<class ArgR,class ArgW,class Function>
Function TasksDispach::sub_run_specx_RW(ArgR myArgR,ArgW myArgW,Function myFunc)
{
        auto begin = std::chrono::steady_clock::now();
        SpRuntime runtime(nbTh);  
        nbTh= runtime.getNbThreads();
        int iValue=0;
        for(int k= 0; k < nbTh; ++k)
        { 
            if (qInfo) { std::cout<<"Call num Thread Read Specx="<<k<<"\n"; }
            runtime.task(SpRead(myArgR),SpWrite(myArgW),myFunc).setTaskName("Op("+std::to_string(k)+")");
            //usleep(1);
            std::atomic_int counter(0);
        }
        runtime.waitAllTasks();
        runtime.stopAllThreads();
        auto end = std::chrono::steady_clock::now(); 
        if (qSave)
        {
            runtime.generateDot(FileName+".dot", true);
            runtime.generateTrace(FileName+".svg");   
        }
              
        if (qInfo) { std::cout<<"\n"; }
        if (qViewChrono) {  std::cout << "===> Elapsed microseconds: "<< std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()<< " us\n"; std::cout<<"\n"; }
    return myFunc;
}




template<class Function>
Function TasksDispach::run(Function myFunc)
{
  switch(numTypeTh) {
    case 1: return(sub_run_async(myFunc));
    break;
    case 2: return(sub_run_specx_R(myFunc));
    break;
    case 3: return(sub_run_specx_W(myFunc));
    break;
    default:
        return(sub_run_multithread(myFunc));
  }
}