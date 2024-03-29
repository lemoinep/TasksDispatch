
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

#include "napp.hpp"


//=======================================================================================================================
// Meta function tools allowing you to process an expression defined in Task
//=======================================================================================================================

constexpr auto& _parameters = NA::identifier<struct parameters_tag>;
constexpr auto& _task = NA::identifier<struct task_tag>;

namespace Backend{

    template<typename ...T, size_t... I>
    auto extractParametersAsTuple( std::tuple<T...> && t, std::index_sequence<I...>)
    {
        return std::forward_as_tuple( std::get<I>(t).getValue()...);
    }

    struct Runtime{
        template <typename ... Ts>
        void task(Ts && ... ts ) {
            auto t = std::make_tuple( std::forward<Ts>(ts)... );
            auto callback = std::get<sizeof...(Ts) - 1>(t);
            auto parameters = extractParametersAsTuple( std::move(t), std::make_index_sequence<sizeof...(Ts)-1>{} );
            std::apply( callback, std::move(parameters) );
        }
    };

    template <typename T,bool b>
    class SpData
    {
        static_assert(std::is_reference<T>::value,
                    "The given type must be a reference");
    public:
        using value_type = T;
        static constexpr bool isWrite = b;

        template <typename U, typename = std::enable_if_t<std::is_convertible_v<U,T>> >
        constexpr explicit SpData( U && u ) : M_val( std::forward<U>(u) ) {}

        constexpr value_type getValue() { return M_val; }
    private:
        value_type M_val;
    };

    template <typename T>
    auto spRead( T && t )
    {
        return SpData<T,false>{ std::forward<T>( t ) };
    }
    template <typename T>
    auto spWrite( T && t )
    {
        return SpData<T,true>{ std::forward<T>( t ) };
    }

    template<typename T>
    auto toSpData( T && t )
    {
        if constexpr ( std::is_const_v<std::remove_reference_t<T>> )
            return spRead( std::forward<T>( t ) );
        else
            return spWrite( std::forward<T>( t ) );
    }

    template<typename ...T, size_t... I>
    auto makeSpDataHelper( std::tuple<T...>& t, std::index_sequence<I...>)
    {
        return std::make_tuple( toSpData(std::get<I>(t))...);
    }
    template<typename ...T>
    auto makeSpData( std::tuple<T...>& t ){
        return makeSpDataHelper<T...>(t, std::make_index_sequence<sizeof...(T)>{});
    }

    template<typename T>
    auto toSpDataSpecx( T && t )
    {
        if constexpr ( std::is_const_v<std::remove_reference_t<T>> )
            return SpRead(std::forward<T>( t ));
        else
            return SpWrite(std::forward<T>( t ));
    }

    template<typename ...T, size_t... I>
    auto makeSpDataHelperSpecx( std::tuple<T...>& t, std::index_sequence<I...>)
    {
        return std::make_tuple( toSpDataSpecx(std::get<I>(t))...);
    }
    template<typename ...T>
    auto makeSpDataSpecx( std::tuple<T...>& t ){
        return makeSpDataHelperSpecx<T...>(t, std::make_index_sequence<sizeof...(T)>{});
    }
}



namespace Frontend
{
/*
    template <typename ... Ts>
    void
    runTask( Ts && ... ts )
    {
        auto args = NA::make_arguments( std::forward<Ts>(ts)... );
        auto && task = args.get(_task);
        auto && parameters = args.get_else(_parameters,std::make_tuple());
        Backend::Runtime runtime;

        std::apply( [&runtime](auto... args){ runtime.task(args...); }, std::tuple_cat( Backend::makeSpData( parameters ), std::make_tuple( task ) ) );
    }
*/

    template <typename ... Ts>
    auto parameters(Ts && ... ts)
    {
        //Construit un tuple de références aux arguments dans args pouvant être transmis en tant qu'argument à une fonction
        return std::forward_as_tuple( std::forward<Ts>(ts)... );
    }
}




//================================================================================================================================
// CLASS TASKsDISPATCH: Provide a family of multithreaded functions...
//================================================================================================================================

// Nota: The objective is to provide a range of tools in the case of using a single variable in multithreading.
// In the case of work with several variables use the class TasksDispatchComplex.


void *WorkerInNumCPU(void *arg) {
    std::function<void()> *func = (std::function<void()>*)arg;
    (*func)();
    pthread_exit(NULL);
}


class TasksDispatch
{
    private:
        int nbThTotal;   
        std::vector<int> indice;
        void initIndice();

    public:
         //BEGIN::Small functions and variables to manage initialization parameters
        int  numTypeTh;
        int  nbTh;
        bool qSave;
        bool qSlipt;
        bool qDeferred;
        bool qViewChrono;
        bool qInfo;
        std::string FileName;
       
        auto begin(); 
        auto end(); 
        auto size(); 

        int  getNbMaxThread();
        void init(int numType,int nbThread,bool qsaveInfo);
        void setFileName(std::string s);
        void setNbThread(int v);
         //END::Small functions and variables to manage initialization parameters

        //BEGIN::multithread with std::ansync or Specx or ... part
        template<class Function>
            Function run(Function myFunc);
                template<class Function>
                    Function sub_run_multithread(Function myFunc);
                template<class Function>
                    Function sub_run_async(Function myFunc);
                template<class Function>
                    Function sub_run_specx(Function myFunc);

        template<class Function>
            std::vector<double> run_beta(Function myFunc);
                template<class Function>
                    std::vector<double> sub_run_multithread_beta(Function myFunc);
                template<class Function> 
                    std::vector<double> sub_run_specx_beta(Function myFunc);
                template<class Function>
                    std::vector<double> sub_run_async_beta(Function myFunc);
        //END::multithread  part

        //BEGIN::Detach part
        template<typename FctDetach>
                    auto sub_detach_future_alpha(FctDetach&& func) -> std::future<decltype(func())>;
        template<typename FctDetach>
                    auto sub_detach_future_beta(FctDetach&& func) -> std::future<decltype(func())>;
        template <typename result_type, typename FctDetach>
                    std::future<result_type> sub_detach_future_gamma(FctDetach func);

        template<class FunctionLambda,class FunctionLambdaDetach>
                    void sub_detach_specx_beta(FunctionLambda myFunc,int nbThreadsA,FunctionLambdaDetach myFuncDetach,int nbThreadsD);
        //END::Detach part

        //BEGIN::Thread affinity part
        template<class Function>
            void RunTaskInNumCPU(int idCPU,Function myFunc);
        template<class Function>
            void RunTaskInNumCPUs(const std::vector<int> & numCPU ,Function myFunc);
        //END::Thread affinity part

        template<class InputIterator, class Function>
            Function for_each(InputIterator first, InputIterator last,Function myFunc);

        TasksDispatch(void);
        ~TasksDispatch(void);
};


TasksDispatch::TasksDispatch() { 
    nbThTotal=std::thread::hardware_concurrency();
    nbTh=nbThTotal;
    numTypeTh=2; 
    qSave=false;
    FileName="TestDispatch";
    qSlipt=false;
    qDeferred=false;
    qViewChrono=true;
    qViewChrono=false;
    qInfo=false;  
    initIndice();
}

TasksDispatch::~TasksDispatch(void) { 
    //... must be defined
}




auto TasksDispatch::begin()
{
    return(indice.begin());
}

auto TasksDispatch::end()
{
    return(indice.end());
}

auto TasksDispatch::size()
{
    return(indice.size());
}

void TasksDispatch::initIndice()
{
    indice.clear(); for (int i = 1; i <= nbTh; ++i)  { indice.push_back(i); }
}

void TasksDispatch::setNbThread(int v)
{
    nbTh=std::min(v,nbThTotal);
    initIndice();
}

void TasksDispatch::init(int numType,int nbThread,bool qsaveInfo)
{
    numTypeTh=numType; nbTh=nbThread; qSave=qsaveInfo; qInfo=false;  
    initIndice();
}


void TasksDispatch::setFileName(std::string s)
{
    FileName=s;
}

int TasksDispatch::getNbMaxThread()
{
    nbThTotal=std::thread::hardware_concurrency();
    return(nbThTotal);
}


template<class Function>
void TasksDispatch::RunTaskInNumCPU(int idCPU,Function myFunc)
{
    const std::vector<int> v={idCPU};
    RunTaskInNumCPUs(v,myFunc);
}

template<class Function>
void TasksDispatch::RunTaskInNumCPUs(const std::vector<int> & numCPU ,Function myFunc)
{
  int nbTh=numCPU.size();
  std::function<void()> func =myFunc;
  pthread_t thread_array[nbTh];
  pthread_attr_t pta_array[nbTh];

  for (int i = 0; i < nbTh; i++) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(numCPU[i], &cpuset);
    std::cout<<"Num CPU="<< numCPU[i] <<" activated"<<std::endl;
    pthread_attr_init(&pta_array[i]);
    pthread_attr_setaffinity_np(&pta_array[i], sizeof(cpuset), &cpuset);
    if (pthread_create(&thread_array[i],&pta_array[i],WorkerInNumCPU,&func)) { std::cerr << "Error in creating thread" << std::endl; }
  }

  for (int i = 0; i < nbTh; i++) {
        pthread_join(thread_array[i], NULL);
  }

  for (int i = 0; i < nbTh; i++) {
        pthread_attr_destroy(&pta_array[i]);
  }
}


template<typename FctDetach>
auto TasksDispatch::sub_detach_future_alpha(FctDetach&& func) -> std::future<decltype(func())>
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
auto TasksDispatch::sub_detach_future_beta(FctDetach&& func) -> std::future<decltype(func())>
{
    auto task   = std::packaged_task<decltype(func())()>(std::forward<FctDetach>(func));
    auto future = task.get_future();
    std::thread(std::move(task)).detach();
    return std::move(future);
}


template <typename result_type, typename FctDetach>
    std::future<result_type> TasksDispatch::sub_detach_future_gamma(FctDetach func) {
    std::promise<result_type> pro;
    std::future<result_type> fut = pro.get_future();
    std::thread([func](std::promise<result_type> p){p.set_value(func());},std::move(pro)).detach();
    return fut;
}

template<class FunctionLambda,class FunctionLambdaDetach>
void TasksDispatch::sub_detach_specx_beta(FunctionLambda myFunc,int nbThreadsA,FunctionLambdaDetach myFuncDetach,int nbThreadsD)
{ 
    if (qInfo) { std::cout<<"Call Specx Detach="<<"\n"; }
    SpRuntime runtimeA(nbThreadsA);
    SpRuntime runtimeD(nbThreadsD);
    int idData=1;
    runtimeA.task(SpWrite(idData),
        [&,&runtimeD](int & depFakeData)
        {
            runtimeD.task([&,&depFakeData]()
            {
                myFuncDetach();
            });
            myFunc();
        });
    runtimeA.waitAllTasks();
    runtimeA.stopAllThreads();
    if (qSave)
    {
        runtimeA.generateDot("DetachA.dot", true);
        runtimeA.generateTrace("DetachA.svg");   
    }
    runtimeD.waitAllTasks();
    runtimeD.stopAllThreads();
    if (qSave)
    {
        runtimeD.generateDot("DetachD.dot", true);
        runtimeD.generateTrace("DetachD.svg");   
    }
    if (qInfo) { std::cout << std::endl; }
} 


template<class InputIterator, class Function>
Function TasksDispatch::for_each(InputIterator first, InputIterator last,Function myFunc)
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
Function TasksDispatch::sub_run_multithread(Function myFunc)
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
std::vector<double> TasksDispatch::sub_run_multithread_beta(Function myFunc)
{
        std::vector<double> valuesVec(nbTh,0);
        auto begin = std::chrono::steady_clock::now();
        auto LF=[&](const int& k) {  myFunc(k,valuesVec.at(k)); return true;};
        std::vector<std::thread> mythreads;
        for(int k= 0; k < nbTh; ++k){ 
            auto const& idk = k;
            if (qInfo) { std::cout<<"Call num Multithread ="<<k<<"\n"; }
            std::thread th(LF,idk);
            mythreads.push_back(move(th));
        }
        for (std::thread &t : mythreads) {
            t.join();
        }
        auto end = std::chrono::steady_clock::now();
        if (qInfo) { std::cout<<"\n"; }
        if (qViewChrono) {  std::cout << "===> Elapsed microseconds: "<< std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()<< " us\n"; std::cout<<"\n"; }
    return valuesVec;
}


template<class Function>
Function TasksDispatch::sub_run_async(Function myFunc)
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
std::vector<double> TasksDispatch::sub_run_async_beta(Function myFunc)
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
std::vector<double> TasksDispatch::sub_run_specx_beta(Function myFunc)
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
Function TasksDispatch::sub_run_specx(Function myFunc)
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
        if (qViewChrono) {  std::cout << "===> Elapsed microseconds: "<< std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()<< " us\n"; std::cout<<"\n"; }
    return myFunc;
}


template<class Function>
Function TasksDispatch::run(Function myFunc)
{
  switch(numTypeTh) {
    case 1: return(sub_run_async(myFunc));
    break;
    case 2: return(sub_run_specx(myFunc));
    break;
    default:
        return(sub_run_multithread(myFunc));
  }
}

template<class Function>
std::vector<double> TasksDispatch::run_beta(Function myFunc)
{
  switch(numTypeTh) {
    case 1: return(sub_run_async_beta(myFunc));
    break;
    case 2: return(sub_run_specx_beta(myFunc));
    break;
    default:
        return(sub_run_multithread_beta(myFunc));
  }
}


//================================================================================================================================
// CLASS TASKsDISPATCH Complex
//================================================================================================================================

class TasksDispatchComplex 
{
    private:
        int nbThTotal;   
        std::string FileName;
        template <typename ... Ts>
        auto parameters(Ts && ... ts);

    public:
        //BEGIN::Small functions and variables to manage initialization parameters
        int nbTh;
        int numTypeTh;
        bool qViewChrono;
        bool qInfo;
        bool qSave;
        bool qDeferred;
        bool qUseIndex;

        void setNbThread(int v);
        int  getNbMaxThread();
        void setFileName(std::string s);
        //END::Small functions and variables to manage initialization parameters

        template <typename ... Ts>
            void runTaskSimple( Ts && ... ts );

        template <typename ... Ts>
            void runTaskSimpleSpecx( Ts && ... ts );

        //BEGIN::multithread with std::ansync or Specx or ... part
        template <typename ... Ts>
            void run( Ts && ... ts );

            template <typename ... Ts>
                void sub_runTaskLoopAsync( Ts && ... ts );

            template <typename ... Ts>
                void sub_runTaskLoopSpecx( Ts && ... ts );
        //END::multithread with std::ansync or Specx or ... part

        //BEGIN::Thread affinity part
        template <typename ... Ts>
        void RunTaskInNumCPUs(const std::vector<int> & numCPU,Ts && ... ts);
        //END::Thread affinity part
        
        TasksDispatchComplex(void);
        ~TasksDispatchComplex(void);
};


TasksDispatchComplex::TasksDispatchComplex()
{
    nbThTotal=std::thread::hardware_concurrency();
    nbTh=nbThTotal;
    qViewChrono=false;
    qInfo=false;
    qSave=false;
    qDeferred=true;
    numTypeTh=0;
    qUseIndex=false;
    FileName="TestDispatchComplex";
}

TasksDispatchComplex::~TasksDispatchComplex()
{
    //... must be defined
}



void TasksDispatchComplex::setFileName(std::string s)
{
    FileName=s;
}

void TasksDispatchComplex::setNbThread(int v)
{
    nbTh=std::min(v,nbThTotal);
}

int TasksDispatchComplex::getNbMaxThread()
{
    nbThTotal=std::thread::hardware_concurrency();
    return(nbThTotal);
}

template <typename ... Ts>
auto TasksDispatchComplex::parameters(Ts && ... ts)
{
    return std::forward_as_tuple( std::forward<Ts>(ts)... );
}





template <typename ... Ts>
void TasksDispatchComplex::runTaskSimple( Ts && ... ts )
{
    auto begin = std::chrono::steady_clock::now();
    auto args = NA::make_arguments( std::forward<Ts>(ts)... );
    auto && task = args.get(_task);
    auto && parameters = args.get_else(_parameters,std::make_tuple());
    Backend::Runtime runtime;
    auto tp=std::tuple_cat( 
					Backend::makeSpData( parameters ), 
					std::make_tuple( task ) 
				);
    std::apply( [&runtime](auto... args){ runtime.task(args...); }, tp );
    auto end = std::chrono::steady_clock::now();
    if (qViewChrono) {  std::cout << "===> Elapsed microseconds: "<< std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()<< " us\n"; std::cout<<"\n"; }
}

template <typename ... Ts>
void TasksDispatchComplex::runTaskSimpleSpecx( Ts && ... ts )
{
    auto begin = std::chrono::steady_clock::now();
    auto args = NA::make_arguments( std::forward<Ts>(ts)... );
    auto && task = args.get(_task);
    auto && parameters = args.get_else(_parameters,std::make_tuple());
    Backend::Runtime runtime;
    std::cout <<"Specx nbTh="<<nbTh<< "\n";
    SpRuntime runtime_Specx(nbTh); 
    auto tpBackend=Backend::makeSpDataSpecx( parameters );
         int NbtpBackend=std::tuple_size<decltype(tpBackend)>::value;
        std::cout <<"Size tpBackend="<<NbtpBackend<< std::endl;

    auto tpSpecx=std::tuple_cat( 
					Backend::makeSpDataSpecx( parameters ), 
					std::make_tuple( task ) 
				);

    std::apply([&](auto &&... args) { runtime_Specx.task(args...); },tpSpecx);
    runtime_Specx.waitAllTasks();
    runtime_Specx.stopAllThreads();
    auto end = std::chrono::steady_clock::now();
    if (qViewChrono) {  std::cout << "===> Elapsed microseconds: "<< std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()<< " us\n"; std::cout<<"\n"; }
}

template <typename ... Ts>
void TasksDispatchComplex::sub_runTaskLoopAsync( Ts && ... ts )
{
    auto begin = std::chrono::steady_clock::now();
    auto args = NA::make_arguments( std::forward<Ts>(ts)... );
    auto && task = args.get(_task);
    auto && parameters = args.get_else(_parameters,std::make_tuple());
    Backend::Runtime runtime;
    std::vector< std::future< bool > > futures;
    std::cout <<"std::Async nbTh="<<nbTh<< "\n";

    for (int k = 0; k < nbTh; k++) {
        if (qInfo) { std::cout<<"Call num Thread futures="<<k<<"\n"; }

        if (qUseIndex) { std::get<0>(parameters)=std::cref(k); /*std::cout<<"k="<<std::get<0>(parameters)<<"\n";*/ }
		auto tp=std::tuple_cat( 
					Backend::makeSpData( parameters ), 
					std::make_tuple( task ) 
				);

		auto LamdaTransfert = [&]() {
			std::apply([&runtime](auto... args){ runtime.task(args...); }, tp);
            return true; 
		};

        if (qDeferred)
        {
            futures.emplace_back(
            std::async(std::launch::deferred,LamdaTransfert));
        }
        else
        {
            futures.emplace_back(
                std::async(std::launch::async,LamdaTransfert));
        }
    }

    for( auto& r : futures){ auto a =  r.get(); }
    auto end = std::chrono::steady_clock::now();
    if (qViewChrono) {  std::cout << "===> Elapsed microseconds: "<< std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()<< " us\n"; std::cout<<"\n"; }
}

/*
template <typename ... Ts>
void TasksDispatchComplex::sub_runTaskLoopMultithread( Ts && ... ts )
{
    //add mutex
    auto begin = std::chrono::steady_clock::now();
    auto args = NA::make_arguments( std::forward<Ts>(ts)... );
    auto && task = args.get(_task);
    auto && parameters = args.get_else(_parameters,std::make_tuple());
    Backend::Runtime runtime;
    std::vector<std::thread> mythreads;
    std::cout <<"nbTh="<<nbTh<< "\n";
    std::mutex mtx; 
    for (int k = 0; k < nbTh; k++) {
        std::cout<<"Call num Thread futures="<<k<<"\n";
		auto tp=std::tuple_cat( 
					Backend::makeSpData( parameters ), 
					std::make_tuple( task ) 
				);
		auto LamdaTransfert = [&]() {
			std::apply([&runtime](auto... args){ runtime.task(args...); }, tp);
            return true; 
		};
        std::thread th(LamdaTransfert);
        mythreads.push_back(move(th));
    }
    for (std::thread &t : mythreads) { t.join();}
    auto end = std::chrono::steady_clock::now();
    if (qViewChrono) {  std::cout << "===> Elapsed microseconds: "<< std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()<< " us\n"; std::cout<<"\n"; }
}
*/


template <typename ... Ts>
void TasksDispatchComplex::sub_runTaskLoopSpecx( Ts && ... ts )
{
    auto begin = std::chrono::steady_clock::now();
    auto args = NA::make_arguments( std::forward<Ts>(ts)... );
    auto && task = args.get(_task);
    auto && parameters = args.get_else(_parameters,std::make_tuple());
    Backend::Runtime runtime;
    std::cout <<"Specx nbTh="<<nbTh<< "\n";

    SpRuntime runtime_Specx(nbTh); 
   
    auto tpBackend=Backend::makeSpDataSpecx( parameters );
         int NbtpBackend=std::tuple_size<decltype(tpBackend)>::value;
        std::cout <<"Size tpBackend="<<NbtpBackend<< std::endl;

    auto tpSpecx=std::tuple_cat( 
					Backend::makeSpDataSpecx( parameters ), 
					std::make_tuple( task ) 
				);

    for (int k = 0; k < nbTh; k++) {
        if (qInfo) { std::cout<<"Call num Thread specx="<<k<<"\n"; }
        std::apply([&](auto &&... args) { runtime_Specx.task(args...); },tpSpecx);
    }

    runtime_Specx.waitAllTasks();
    runtime_Specx.stopAllThreads();

    auto end = std::chrono::steady_clock::now();

    if (qSave)
    {
        runtime_Specx.generateDot(FileName+".dot", true);
        runtime_Specx.generateTrace(FileName+".svg");  
    }

    if (qViewChrono) {  std::cout << "===> Elapsed microseconds: "<< std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()<< " us\n"; std::cout<<"\n"; }
}


template <typename ... Ts>
void TasksDispatchComplex::run( Ts && ... ts )
{
    switch(numTypeTh) {
        case 1: sub_runTaskLoopAsync(std::forward<Ts>(ts)...);
        break;
        case 2: sub_runTaskLoopSpecx(std::forward<Ts>(ts)...);
        break;
        default: sub_runTaskLoopAsync(std::forward<Ts>(ts)...);
    }
}


template <typename ... Ts>
void TasksDispatchComplex::RunTaskInNumCPUs(const std::vector<int> & numCPU,Ts && ... ts)
{
  int nbTh=numCPU.size();
  pthread_t thread_array[nbTh];
  pthread_attr_t pta_array[nbTh];

  auto begin = std::chrono::steady_clock::now();
  auto args = NA::make_arguments( std::forward<Ts>(ts)... );
  auto && task = args.get(_task);
  auto && parameters = args.get_else(_parameters,std::make_tuple());
  Backend::Runtime runtime;

  qUseIndex=true;
  
  for (int i = 0; i < nbTh; i++) {
    int const& idk = i;
    if (qUseIndex) { std::get<0>(parameters)=idk; }
    auto tp=std::tuple_cat( 
            Backend::makeSpData( parameters ), 
                      std::make_tuple( task ) 
    );

    auto LamdaTransfert = [&]() {
                std::apply([&runtime](auto... args){ runtime.task(args...); }, tp);
                return true; 
    };
    std::function<void()> func =LamdaTransfert;
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(numCPU[i], &cpuset);
    std::cout<<"Num CPU="<< numCPU[i] <<" activated"<<std::endl;
    pthread_attr_init(&pta_array[i]);
    pthread_attr_setaffinity_np(&pta_array[i], sizeof(cpuset), &cpuset);
    if (pthread_create(&thread_array[i],&pta_array[i],WorkerInNumCPU,&func)) { std::cerr << "Error in creating thread" << std::endl; }
  }

  for (int i = 0; i < nbTh; i++) {
        pthread_join(thread_array[i], NULL);
  }

  for (int i = 0; i < nbTh; i++) {
        pthread_attr_destroy(&pta_array[i]);
  }
}


//================================================================================================================================
// CLASS TASKsDISPATCH Complex Beta
//================================================================================================================================

class TasksDispatchComplexBeta
{
    private:
        int nbThTotal;   
        std::vector<int> indice;
        void initIndice();
        std::string FileName;
        template <typename ... Ts>
        auto parameters(Ts && ... ts);
    public:
        //BEGIN::Small functions and variables to manage initialization parameters
        int nbTh;
        int numTypeTh;
        bool qViewChrono;
        bool qInfo;
        bool qSave;
        bool qDeferred;
        bool qUseIndex;

        void setNbThread(int v);
        int  getNbMaxThread();
        void setFileName(std::string s);
        //END::Small functions and variables to manage initialization parameters
        TasksDispatchComplexBeta(void);
        ~TasksDispatchComplexBeta(void);
};

TasksDispatchComplexBeta::TasksDispatchComplexBeta()
{
    nbThTotal=std::thread::hardware_concurrency();
    nbTh=nbThTotal;
    qViewChrono=false;
    qInfo=false;
    qSave=false;
    qDeferred=true;
    numTypeTh=0;
    qUseIndex=false;
    FileName="NoName";
}

TasksDispatchComplexBeta::~TasksDispatchComplexBeta()
{
    //... must be defined
}

void TasksDispatchComplexBeta::setFileName(std::string s)
{
    FileName=s;
}

void TasksDispatchComplexBeta::setNbThread(int v)
{
    nbTh=std::min(v,nbThTotal);
    initIndice();
}

int TasksDispatchComplexBeta::getNbMaxThread()
{
    nbThTotal=std::thread::hardware_concurrency();
    return(nbThTotal);
}

void TasksDispatchComplexBeta::initIndice()
{
    indice.clear(); for (int i = 1; i <= nbTh; ++i)  { indice.push_back(i); }
}

template <typename ... Ts>
auto TasksDispatchComplexBeta::parameters(Ts && ... ts)
{
    return std::forward_as_tuple( std::forward<Ts>(ts)... );
}




//================================================================================================================================
// TOOLS: A panel of functions allowing you to control the functionality of class TasksDispatch and TasksDispatchComplex
//================================================================================================================================

void testScanAllThreadMethods()
{
    int qPutLittleTroublemaker=true;
    int time_sleep= 100000;
    auto P001=[time_sleep,qPutLittleTroublemaker](const int i,double& s) {  
            double sum=0.0; 
            for(int j=0;j<100;j++) { sum+=double(j); }
            if (qPutLittleTroublemaker) {
                srand((unsigned)time(0)); int time_sleep2 = rand() % 5000 + 1; usleep(time_sleep2); 
            }
            usleep(time_sleep);
            s=sum+i;      
        return true;
    };

    bool qChrono=false;

    TasksDispatch Fg1; 
    int nbThreads = Fg1.nbTh;
    //int nbThreads = 96;
    Color(7); std::cout<<"Test Scan [";
    Color(3); std::cout<<nbThreads;
    Color(7); std::cout<<"] Thread Methods >>> ";
    Fg1.setFileName("Results"); 
    Fg1.init(0,nbThreads,true); Fg1.qViewChrono=qChrono; 
    //std::vector<double> valuesVec1=Fg1.sub_run_multithread_beta(P001);
    std::vector<double> valuesVec1=Fg1.run_beta(P001);
    double Value1=std::reduce(valuesVec1.begin(),valuesVec1.end()); 

    TasksDispatch Fg2; 
    Fg2.setFileName("Results"); 
    Fg2.init(1,nbThreads,true); Fg2.qViewChrono=qChrono; 
    //std::vector<double> valuesVec2=Fg2.sub_run_async_beta(P001);
    std::vector<double> valuesVec2=Fg1.run_beta(P001);
    double Value2=std::reduce(valuesVec2.begin(),valuesVec2.end()); 

    TasksDispatch Fg3; 
    Fg3.setFileName("Results"); 
    Fg3.init(2,nbThreads,true); Fg3.qViewChrono=qChrono; 
    //std::vector<double> valuesVec3=Fg3.sub_run_specx_beta(P001);
    std::vector<double> valuesVec3=Fg1.run_beta(P001);
    double Value3=std::reduce(valuesVec3.begin(),valuesVec3.end()); 
    if ((Value1==Value2) && (Value1==Value3)) {
        Color(2); std::cout <<"OK"<< "\n"; 
    } 
    else 
    {
        Color(1); std::cout <<"ERROR "<<"m1:"<<Value1<<" m2:"<<Value2<<" m3:"<<Value3<< "\n"; 
    }
    std::cout << "\n"; 
    Color(7);
    std::cout << "\n"; 
}


//================================================================================================================================
// 
//================================================================================================================================


