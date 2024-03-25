
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
#include "SpComputeEngine.hpp"
#include "Speculation/SpSpeculativeModel.hpp"

#include "napp.hpp"


//#define COMPILE_WITH_CUDA
//#define COMPILE_WITH_HIP

#define MODE_NO_THREAD 0
#define MODE_THREAD 1
#define MODE_ASYNC 2
#define MODE_SPECX 3


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


class TiT
{
    private:
        int nbThTotal;   
        std::string FileName;
        template <typename ... Ts>
        auto parameters(Ts && ... ts);
        SpTaskGraph<SpSpeculativeModel::SP_NO_SPEC> mytg;
        SpComputeEngine myce;

        std::vector<int> numTaskStatus;
        
        std::vector<std::future<bool>> myfutures;
        std::vector<std::thread> mythreads;
        std::vector<pthread_t> mypthread_t; 
        //pthread_t mypthread_t[100]; 
        pthread_attr_t mypthread_attr_t;
        std::vector<int> mypthread_cpu; 
        std::mutex mymtx;
        std::promise<int> promise0;
        std::chrono::steady_clock::time_point t_begin,t_end;
        long int t_laps;
        bool qFirstTask;
        int  idk;
        int  numLevelAction;

        template <typename ... Ts> 
            auto common(Ts && ... ts);
        

    public:
        //BEGIN::Small functions and variables to manage initialization parameters
        int  nbTh;
        int  numTypeTh;
        bool qViewChrono;
        bool qInfo;
        bool qSave;
        bool qDeferred;
        bool qUseIndex;
        bool qDetach;
        bool qCUDA;



        void setNbThread      (int v)   { nbTh=std::min(v,nbThTotal); }
        int  getNbMaxThread   ()        { nbThTotal=std::thread::hardware_concurrency(); return(nbThTotal); }
        int  getNbThreads     () const  { int val=nbTh; if (numTypeTh==3) { val=static_cast<int>(myce.getCurrentNbOfWorkers()); } return val; }
        int  getNbCpuWorkers  () const  { int val=nbTh; if (numTypeTh==3) { val=static_cast<int>(myce.getNbCpuWorkers()); } return val; }

        long int  getTimeLaps ()        { return t_laps; }

        #ifdef COMPILE_WITH_CUDA
            int getNbCudaWorkers() const {
                return static_cast<int>(myce.getNbCudaWorkers());
                }
        #endif
        #ifdef COMPILE_WITH_HIP
            int getNbHipWorkers() const {
                return static_cast<int>(myce.getNbHipWorkers());
            }
        #endif
            
        void setFileName(std::string s) { FileName=s; }
        //END::Small functions and variables to manage initialization parameters


        TiT(void);
        ~TiT(void);

        explicit TiT(const int nbThread,int numTypeThread):mytg(),myce(SpWorkerTeamBuilder::TeamOfCpuWorkers(nbThread))
        {
            nbTh=nbThread;
            numTypeTh=numTypeThread;
            idk=0; numTaskStatus.clear(); qDetach=0;
            t_begin = std::chrono::steady_clock::now();
            
            
            if (numTypeTh==0) { } //No Thread
            if (numTypeTh==1) { } //multithread
            if (numTypeTh==2) { } //std::async
            if (numTypeTh==3) { mytg.computeOn(myce); }

            if (numTypeTh==10) { pthread_attr_init(&mypthread_attr_t); }
            

            if (qInfo)
            {
                if (numTypeTh==0) { std::cout<<"[INFO]: Mode No Thread\n"; }
                if (numTypeTh==1) { std::cout<<"[INFO]: Mode Multithread\n"; }
                if (numTypeTh==2) { std::cout<<"[INFO]: Mode Std::async\n"; }
                if (numTypeTh==3) { std::cout<<"[INFO]: Mode Specx\n"; }

                if (numTypeTh==10) { std::cout<<"[INFO]: Mode Thread in CPU\n"; }
            }
            
        }
        
        #ifdef COMPILE_WITH_CUDA
            TiT() :
                mytg(), myce(SpWorkerTeamBuilder::TeamOfCpuCudaWorkers()) {
            mytg.computeOn(myce);
        }
        #endif

        #ifdef COMPILE_WITH_HIP
            TiT() :
                mytg(), myce(SpWorkerTeamBuilder::TeamOfCpuHipWorkers()) {
            mytg.computeOn(myce);
        }
        #endif

        

        template <class ClassFunc> void execOnWorkers(ClassFunc&& func) { myce.execOnWorkers(std::forward<ClassFunc>(func)); }
  
        template <typename ... Ts>
        void add( Ts && ... ts );

            template <typename ... Ts>
                void addTaskSimple( Ts && ... ts );

            template <typename ... Ts>
                void addTaskSpecx( Ts && ... ts );

            template <typename ... Ts>
                void addTaskAsync( Ts && ... ts );

            template <typename ... Ts>
                void addTaskMultithread( Ts && ... ts );


        template <class InputIterator,typename ... Ts>
            void for_each(InputIterator first, InputIterator last,Ts && ... ts);


        template <typename ... Ts>
            void add(int numCPU,Ts && ... ts);


        template<typename FctDetach>
            auto add_detach(FctDetach&& func) -> std::future<decltype(func())>;

        template <typename ... Ts>
            void runInCPUs(const std::vector<int> & numCPU,Ts && ... ts);
        
        void run();
        void close();
        void debriefingTasks();


};



TiT::TiT()
{
    nbThTotal=std::thread::hardware_concurrency();
    nbTh=nbThTotal;
    qInfo=false;
    qInfo=true;
    qSave=false;
    qDeferred=false;
    numTypeTh=0;
    qUseIndex=false;
    FileName="NoName";
    qFirstTask=true;
    idk=0;
    numLevelAction=0;
}

TiT::~TiT()
{
    //Add somes   
    if ((numTypeTh==3) && (numLevelAction==3)) {  myce.stopIfNotAlreadyStopped(); } 
    //Specx
}





template <typename ... Ts>
auto TiT::parameters(Ts && ... ts)
{
    return std::forward_as_tuple( std::forward<Ts>(ts)... );
}

template <typename ... Ts>
    auto TiT::common(Ts && ... ts)
{
    auto args = NA::make_arguments( std::forward<Ts>(ts)... );
    auto && task = args.get(_task);
    auto && parameters = args.get_else(_parameters,std::make_tuple());
    auto tp=std::tuple_cat(  Backend::makeSpData( parameters ), std::make_tuple( task ) );
    return(tp);
}


template <typename ... Ts>
void TiT::addTaskSimple( Ts && ... ts )
{
    auto tp=common(std::forward<Ts>(ts)...);
    Backend::Runtime runtime;
    std::apply( [&runtime](auto... args){ runtime.task(args...); }, tp );
}


template <typename ... Ts>
void TiT::addTaskMultithread( Ts && ... ts )
{
    auto tp=common(std::forward<Ts>(ts)...);
    Backend::Runtime runtime;
	auto LamdaTransfert = [&]() {
			std::apply([&runtime](auto... args){ runtime.task(args...); }, tp);
            return true; 
	};
    std::thread th(LamdaTransfert);
    mythreads.push_back(move(th));
    usleep(1);
}


template <typename ... Ts>
void TiT::addTaskAsync( Ts && ... ts )
{
    auto tp=common(std::forward<Ts>(ts)...);
    Backend::Runtime runtime;
    auto LamdaTransfert = [&]() {
			std::apply([&runtime](auto... args){ runtime.task(args...); }, tp);
            return true; 
		};

    if (qDeferred) { myfutures.emplace_back(std::async(std::launch::deferred,LamdaTransfert));}
    else           { myfutures.emplace_back(std::async(std::launch::async,LamdaTransfert)); }
    usleep(1);
}



template <typename ... Ts>
void TiT::addTaskSpecx( Ts && ... ts )
{
    auto args = NA::make_arguments( std::forward<Ts>(ts)... );
    auto && task = args.get(_task);
    auto && parameters = args.get_else(_parameters,std::make_tuple());
    auto tp=std::tuple_cat( 
					Backend::makeSpDataSpecx( parameters ), 
					std::make_tuple( task ) 
				);
    std::apply([&](auto &&... args) { mytg.task(args...).setTaskName("Op("+std::to_string(idk)+")"); },tp);
    usleep(0); std::atomic_int counter(0);
}


template <typename ... Ts>
void TiT::add( Ts && ... ts )
{
    numLevelAction=1;
    idk++; numTaskStatus.push_back(qDetach);
    if (qFirstTask) { t_begin = std::chrono::steady_clock::now(); qFirstTask=false;}
    //std::cout<<"numTypeTh="<<numTypeTh<<"\n";
    switch(numTypeTh) {
        case 1: addTaskMultithread(std::forward<Ts>(ts)...); //multithread
        break;
        case 2: addTaskAsync(std::forward<Ts>(ts)...); //std::async
        break;
        case 3: addTaskSpecx(std::forward<Ts>(ts)...); //Specx
        break;
        default: addTaskSimple(std::forward<Ts>(ts)...); //No Thread
    }
    qDetach=false;
}


template <class InputIterator,typename ... Ts>
    void TiT::for_each(InputIterator first, InputIterator last,Ts && ... ts)
{
    qUseIndex=true; //Iterator used
    numLevelAction=1;
    if (qFirstTask) { t_begin = std::chrono::steady_clock::now(); qFirstTask=false;}
    for ( ; first!=last; ++first )
    {
        auto const& idk = *first;
        auto args = NA::make_arguments( std::forward<Ts>(ts)... );
        auto && task = args.get(_task);
        auto && parameters = args.get_else(_parameters,std::make_tuple());
        if (qUseIndex) { std::get<0>(parameters)=std::cref(idk); } 
        auto tp=std::tuple_cat( 
					Backend::makeSpData( parameters ), 
					std::make_tuple( task ) 
				);

        Backend::Runtime runtime;

        if (numTypeTh==0) {
            std::apply( [&runtime](auto... args){ runtime.task(args...); }, tp );
        }

        if (numTypeTh==1) {
            auto LamdaTransfert = [&]() {
			    std::apply([&runtime](auto... args){ runtime.task(args...); }, tp);
            return true; 
            };
            std::thread th(LamdaTransfert);
            mythreads.push_back(move(th));
        }

        if (numTypeTh==2) {
            auto LamdaTransfert = [&]() {
			    std::apply([&runtime](auto... args){ runtime.task(args...); }, tp);
            return true; 
		    };

            if (qDeferred) { myfutures.emplace_back(std::async(std::launch::deferred,LamdaTransfert));}
            else           { myfutures.emplace_back(std::async(std::launch::async,LamdaTransfert)); }
            usleep(1);
        }

        if (numTypeTh==3) {
            auto tpSpecx=std::tuple_cat( 
					Backend::makeSpDataSpecx( parameters ), 
					std::make_tuple( task ) 
			);
            std::apply([&](auto &&... args) { mytg.task(args...).setTaskName("Op("+std::to_string(idk)+")"); },tpSpecx);
            usleep(0); std::atomic_int counter(0);
        }
    }

    //run(); 
}


void TiT::run()
{
    numLevelAction=2;
    if (qInfo) { std::cout<<"[INFO]: Run\n"; }
    if (numTypeTh==0) { } //No Thread
    if (numTypeTh==1) { for (std::thread &t : mythreads) { t.join();} } //multithread
    if (numTypeTh==2) { for( auto& r : myfutures){ auto a =  r.get(); }; } //std::async
    if (numTypeTh==3) { promise0.set_value(0); mytg.waitAllTasks(); } //Specx

    if (numTypeTh==10) { 
        for (int i = 0; i < mypthread_cpu.size(); i++) { 
            std::cout<<"[INFO]: Joint "<<i<<"\n";
            pthread_join(mypthread_t[i], NULL); 
        }
        //pthread_attr_destroy(&mypthread_attr_t);
    } //In CPU

    if (qInfo) { std::cout<<"[INFO]: All Tasks Accomplished\n"; }
}

void TiT::close()
{
    numLevelAction=3;
    if (numTypeTh==0) {  } //No Thread
    if (numTypeTh==1) {  } //multithread
    if (numTypeTh==2) {  } //std::async
    if (numTypeTh==3) { myce.stopIfNotAlreadyStopped(); } //Specx
    if (!qFirstTask) { t_end = std::chrono::steady_clock::now(); qFirstTask=true;}
    if (qInfo) { std::cout<<"[INFO]: Close All Tasks annd process\n"; }
    idk=0;
}

void TiT::debriefingTasks()
{
    t_laps=std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_begin).count();

    if (qViewChrono) {  
        if (qInfo) { std::cout << "[INFO]: Elapsed microseconds: "<<t_laps<< " us\n"; }
    }

    if (qSave)
    {
        if (numTypeTh==3) { 
            std::cout << "[INFO]: Save "<<FileName<< "\n";
            mytg.generateDot(FileName+".dot",true); 
            mytg.generateTrace(FileName+".svg",true); 
        }
    }
} 


template<typename FctDetach>
auto TiT::add_detach(FctDetach&& func) -> std::future<decltype(func())>
{
    auto task   = std::packaged_task<decltype(func())()>(std::forward<FctDetach>(func));
    auto future = task.get_future();
    std::thread(std::move(task)).detach();
    return std::move(future);
}

template <typename ... Ts>
void TiT::add(int numCPU,Ts && ... ts)
{
    //Run in num CPU
    mypthread_cpu.push_back(numCPU);
    pthread_t new_thread;
    

    auto args = NA::make_arguments( std::forward<Ts>(ts)... );
    auto && task = args.get(_task);
    auto && parameters = args.get_else(_parameters,std::make_tuple());
    Backend::Runtime runtime;
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
    CPU_SET(numCPU, &cpuset);
    pthread_attr_setaffinity_np(&mypthread_attr_t, sizeof(cpuset), &cpuset);
    int ret=pthread_create(&new_thread,&mypthread_attr_t,WorkerInNumCPU,&func);
    if (ret) { std::cerr << "Error in creating thread" << std::endl; }

    mypthread_t.push_back(new_thread);

    //pthread_attr_destroy(&mypthread_attr_t);

    //vectorOfThreads.resize(NUM_THREADS);
}



template <typename ... Ts>
void TiT::runInCPUs(const std::vector<int> & numCPU,Ts && ... ts)
{
    int nbTh=numCPU.size();
    pthread_t thread_array[nbTh];
    pthread_attr_t pta_array[nbTh];

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
// THE END.
//================================================================================================================================


