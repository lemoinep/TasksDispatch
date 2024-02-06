
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




/*=====================================================================================================*/
/*=====================================================================================================*/




/*=====================================================================================================*/
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
   bool qInfo=true; 
    SpRuntime runtime(6);  
    int nbTh= runtime.getNbThreads();
        int iValue=0;
        for(int k= 0; k < nbTh; ++k)
        { 
            auto const& idk = k;
            if (qInfo) { std::cout<<"Call num Thread Read Specx="<<idk<<"\n"; }
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
    FgCalculIntegral.init(1,nbThreads,true); FgCalculIntegral.setFileName("TestDispachIntegral"); FgCalculIntegral.qViewChrono=true;
    FgCalculIntegral.run(MyAlgo000);
    Color(7);

    std::cout << "\n"; 
    integralValue=h*std::reduce(valuesVec.begin(),valuesVec.end()); 
    std::cout<<"PI Value= "<<integralValue<<"\n";

    /*
    std::cout<<"PI method Specx"<<"\n";
    valuesVec.clear(); std::cout<<"Clear results size="<<valuesVec.size()<< "\n";
    FgCalculIntegral.init(2,nbThreads,true); FgCalculIntegral.setFileName("TestDispachIntegral"); FgCalculIntegral.qViewChrono=true;
    FgCalculIntegral.run(MyAlgo000);
    Color(7);
    std::cout << "\n"; 
    integralValue=h*std::reduce(valuesVec.begin(),valuesVec.end()); 
    std::cout<<"PI Value= "<<integralValue<<"\n"*/
    
}


void activeBlockTest001_Beta(int nbThreads)
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
    Fg1.init(1,nbThreads,true);  Fg1.qInfo=false; Fg1.qViewChrono=true; Fg1.qSave=true; Fg1.setFileName("TestDispachIntegral");
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
    Fg2.init(2,nbThreads,true);  Fg2.qInfo=false; Fg2.qViewChrono=true; Fg2.qSave=true; Fg2.setFileName("TestDispachIntegral");
    std::vector<double> valuesVec2=Fg2.sub_run_specx(MyAlgo000);
    std::cout << "Vec R= "; for (int k=0; k<nbThreads; k++) { Color(k+1); std::cout << valuesVec2[k] << " ";  } 
    std::cout << "\n"; 
    Color(7);
    std::cout << "\n"; 
    integralValue=h*std::reduce(valuesVec2.begin(),valuesVec2.end()); 
    std::cout<<"PI Value= "<<integralValue<<"\n";
    std::cout << "\n"; 
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
    FgCalcul.init(1,nbThreads,true); FgCalcul.setFileName("TestDispachSum");  FgCalcul.qViewChrono=true;
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
    FgCalcul.init(2,nbThreads,true); FgCalcul.setFileName("TestDispachSum"); FgCalcul.qViewChrono=true;    //FgCalcul.qSave=false; 
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

void WriteMat(int Mat[][3],int row,int col)
{
    for(int i = 0; i < row; i++) {
        for(int j = 0; j < col; j++) {
            Color(j+1);
            std::cout << Mat[i][j];
            std::cout << "\t";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
    Color(7);
}

void activeBlockTest003()
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
            //std::cout<<"wValue k="<<k<< std::endl;
            for(int r=0; r<row; ++r)
            {
                int s=0; for(int c=0; c<col; ++c) { s+=MatA[r][c]*MatB[c][k];}
                MatR[r][k]=s;
            }
        return true;
    };

    TasksDispach FgCalcul; 
    FgCalcul.init(1,nbThreads,true); FgCalcul.setFileName("TestDispachMult"); FgCalcul.qViewChrono=true;
    FgCalcul.run(MyAlgo000);

    std::cout<<"MatR=MatA*MatB with STD::ASYNC"<< "\n";
    WriteMat(MatR,row,col);


    FgCalcul.init(2,nbThreads,true); FgCalcul.setFileName("TestDispachMult"); FgCalcul.qViewChrono=true;
    FgCalcul.run(MyAlgo000);

    std::cout<<"MatR=MatA*MatB with Specx"<< "\n";
    WriteMat(MatR,row,col);

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
    Fg1.init(1,12,true); Fg1.setFileName("TestDispachAsync"); //Fg1.qDeferred=true;
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
    Fg5.qSlipt=true;
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


//TEST THREAD WITH FIXED CPU

void *WorkerCPU_Beta(void *param) {
  std::cout<<(char*) param<<std::endl;
  std::cout<<"Hello back form CPU="<< sched_getcpu()<<std::endl;
  //pthread_exit(NULL);
  char *ret;
  strcpy(ret, "Test Output OK");
  pthread_exit(ret);
}

void RunTaskInNumCPU_Beta(int idCPU)
{
  int ret;
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(idCPU, &cpuset);
  pthread_attr_t pta;
  pthread_attr_init(&pta);
  pthread_attr_setaffinity_np(&pta, sizeof(cpuset), &cpuset);

  pthread_t thread;
  char my_thread_msg[] ="Hello send to Thread!";
  if (pthread_create(&thread, &pta,WorkerCPU_Beta,(void*) my_thread_msg) != 0) { std::cerr << "Error in creating thread" << std::endl; }
  char *message;
  ret=pthread_join(thread, (void **)&message);//pthread_join(thread, NULL);
  if(ret != 0) { perror("pthread_join failed"); exit(EXIT_FAILURE);}
  std::cout <<message<< "\n"; 
  pthread_attr_destroy(&pta);
  //pthread_mutex_t if pb 
  pthread_exit(NULL);
}

void RunTaskLoopInSpecifiedCPU(int NbLoop)
{
  constexpr unsigned num_threads = 96;
  // A mutex ensures orderly access to std::cout from multiple threads.
  std::mutex iomutex;
  std::vector<std::thread> threads(num_threads);

      for (unsigned i = 0; i < num_threads; ++i) {
        threads[i] = std::thread(
          [&iomutex, i, NbLoop] {
              std::this_thread::sleep_for(std::chrono::milliseconds(20));
              int iLoop=0; 
              while (iLoop<NbLoop) 
              {
                  iLoop++;
                  {
                    std::lock_guard<std::mutex> iolock(iomutex);
                    std::cout << "Thread #" << i << ": on CPU " << sched_getcpu() <<" NbLoop="<<iLoop<< "\n";
                  }
                  std::this_thread::sleep_for(std::chrono::milliseconds(900));  
              }
              pthread_exit(NULL);
          });

        // Create a cpu_set_t object representing a set of CPUs. Clear it and mark only CPU i as set.
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(i, &cpuset);
        int rc = pthread_setaffinity_np(threads[i].native_handle(),
                                        sizeof(cpu_set_t), &cpuset);
        if (rc != 0) {
          std::cerr << "Error calling pthread_setaffinity_np: " << rc << "\n";
        }
      }
  for (auto& t : threads) { t.join(); }
}


void activeBlock007()
{
    auto MyAlgo000=[&]() {  
        std::cout<<"OK"<< std::endl;
        return true;
    };
    RunTaskLoopInSpecifiedCPU(1);
    std::cout << "\n"<< "\n";
    RunTaskInNumCPU_Beta(2);
    std::cout << "\n"<< "\n"; 
}



void activeBlockTest001B()
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

void activeBlockTestSpecxVector1(int time_sleep)
{
    int nbThreads = 96;
    std::vector<double> valuesVec(nbThreads,0.0);
    auto MyAlgo000=[time_sleep](const int i,double& s) {  
            double sum=0.0; 
            for(int j=0;j<10;j++)
            { 
                sum+=double(j);
            }
            usleep(time_sleep);
            s=sum+i;      
        return true;
    };

    SpRuntime runtime(nbThreads);
    for(int k1 = 0 ; k1 < nbThreads ; ++k1){
        runtime.task(
            SpRead(k1),
            SpWrite(valuesVec.at(k1)),MyAlgo000);
    }
    runtime.waitAllTasks();
    runtime.stopAllThreads();
    runtime.generateDot("Test.dot", true);
    runtime.generateTrace("Test.svg");   
    std::cout << "Vec R= "; for (int k=0; k<nbThreads; k++) { Color(k+1); std::cout << valuesVec[k] << " ";  } 
    std::cout << "\n"; 
    Color(7);
    std::cout << "\n"; 
}

void activeBlockTestSpecxVector2(int time_sleep)
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
    Fg.setFileName("Test"); 
    Fg.init(1,nbThreads,true);  Fg.qInfo=false; Fg.qViewChrono=true; Fg.qSave=true;
    std::vector<double> valuesVec=Fg.sub_run_specx(MyAlgo000);
    std::cout << "Vec R= "; for (int k=0; k<nbThreads; k++) { Color(k+1); std::cout << valuesVec[k] << " ";  } 
    std::cout << "\n"; 
    Color(7);
    std::cout << "\n"; 
}

void activeBlockTestSpecxVector3(int time_sleep)
{
    int nbThreads = 96;
    double integralValue=0.0;
    double R[nbThreads];
    for (int k=0; k<nbThreads; k++) { R[k]=0.0; }

    auto MyAlgo000=[&R,time_sleep](const int& i) {  
            double sum=0.0; 
            for(int j=0;j<100;j++)
            { 
                sum+=double(j);
            }
            usleep(time_sleep);
            R[i]=sum+i;      
        return true;
    };
   
    std::cout<<"With std:async"<< "\n";
    TasksDispach Fg; 
    Fg.setFileName("Test"); 
    Fg.init(1,nbThreads,true);  Fg.qInfo=false;  Fg.qViewChrono=true;
    Fg.run(MyAlgo000);
    std::cout << "Vec R= "; for (int k=0; k<nbThreads; k++) { Color(k+1); std::cout << R[k] << " ";  } 
    std::cout << "\n"; 
    Color(7);
    std::cout << "\n"; 
    std::cout<<"With std:thread"<< "\n";
    Fg.sub_run_multithread(MyAlgo000);
    std::cout << "Vec R= "; for (int k=0; k<nbThreads; k++) { Color(k+1); std::cout << R[k] << " ";  } 
    std::cout << "\n"; 
    Color(7);
    std::cout << "\n"; 
}



void activeBlock008()
{
    //TEST TG
    SpComputeEngine ce(SpWorkerTeamBuilder::TeamOfCpuWorkers());
    SpTaskGraph<SpSpeculativeModel::SP_NO_SPEC> tg;

    tg.computeOn(ce);
    int ValueR=123;
    int ValueW=0;
    std::cout << "ValueW in=" <<ValueW<< std::endl;

    tg.task(SpRead(ValueR),SpWrite(ValueW), [](const int &V1,int &V2){
        V2=456;
        std::cout << "In Specx block" << std::endl;
        usleep(10000);
    }
    );
    tg.waitAllTasks();
    ce.stopIfNotAlreadyStopped();
    std::cout << "ValueW out=" <<ValueW<< std::endl;
    std::cout << "\n"<< "\n"; 
}

void activeBlock008Beta()
{
    //TEST TG-WITHOUT dependencies
    
    SpComputeEngine ce(SpWorkerTeamBuilder::TeamOfCpuWorkers());
    SpTaskGraph<SpSpeculativeModel::SP_NO_SPEC> tg;
    //<SpSpeculativeModel::SP_MODEL_1>

    tg.computeOn(ce);
    int ValueR=123;
    int ValueW=0;
    std::cout << "ValueW in=" <<ValueW<< std::endl;

    tg.task([&ValueW](){
        ValueW=456;
        std::cout << "In Specx block" << std::endl;
        usleep(10000);
        }
    );

    tg.waitAllTasks();
    ce.stopIfNotAlreadyStopped();
    std::cout << "ValueW out=" <<ValueW<< std::endl;
    std::cout << "\n"<< "\n"; 
    
}




int main(int argc, const char** argv) {

  bool qPlayNext=true;
  qPlayNext=false;

// BEGIN::TESTS 
  if (qPlayNext) {
    std::cout << std::endl;
    std::cout << "<<< Block001 >>>" << std::endl;
    activeBlock001();
    std::cout << std::endl;

    std::cout << std::endl;
    std::cout << "<<< Block002 >>>" << std::endl;
    activeBlock002();
    std::cout << std::endl;
  }
// END::TESTS 

  qPlayNext=true;
  //qPlayNext=false;

  // BEGIN::TEST BENCHMARKS
  if (qPlayNext) {
    std::cout << std::endl;
    std::cout << "<<< ====================================== >>>" << std::endl;
    std::cout << "<<< Test calcul intergral  >>>" << std::endl;
    activeBlockTest001();
    std::cout << std::endl;
    activeBlockTest001_Beta(6);
    std::cout << std::endl;

    std::cout << std::endl;
    std::cout << "<<< ====================================== >>>" << std::endl;
    std::cout << "<<< Test calcul sum vector  >>>" << std::endl;
    activeBlockTest002();
    std::cout << std::endl;

    std::cout << std::endl;
    std::cout << "<<< ====================================== >>>" << std::endl;
    std::cout << "<<< Test calcul matrix product  >>>" << std::endl;
    activeBlockTest003();
    std::cout << std::endl;
  }
  // END::TEST BENCHMARKS

  qPlayNext=true;
  //qPlayNext=false;

 // BEGIN::TEST DETACH
  if (qPlayNext) {
    std::cout << std::endl;
    std::cout << "<<< Block004: Detached Future 2 >>>" << std::endl;
    activeBlock004();
    std::cout << std::endl;

    std::cout << std::endl;
    std::cout << "<<< Block005: Deferred >>>" << std::endl;
    activeBlock005();
    std::cout << std::endl;
  }
// END::TEST DETACH


  qPlayNext=true;
  qPlayNext=false;

// BEGIN::TEST NAP
if (qPlayNext) {
    std::cout << std::endl;
    std::cout << "<<< Block006: NAP  >>>" << std::endl;
    //activeBlock006();
    std::cout << std::endl;
}
// END::TEST NAP

  qPlayNext=true;
  qPlayNext=false;

// BEGIN::TEST MULTITHREAD AFFINITY
if (qPlayNext) {
    std::cout << std::endl;
    std::cout << "<<< Block007: Test Multithread Affinity  >>>" << std::endl;
    activeBlock007();
    std::cout << std::endl;
}
// END::TEST MULTITHREAD AFFINITY




  qPlayNext=true;
  //qPlayNext=false;

// BEGIN::TEST VECTORS LIST THREAD
if (qPlayNext) {
    std::cout << std::endl;
    std::cout << "<<< Test Vector  >>>" << std::endl;
    //activeBlockTestSpecxVector1(10000);
    activeBlockTestSpecxVector3(100000);
    std::cout<<"With specx"<< "\n";
    activeBlockTestSpecxVector2(100000);
    std::cout << std::endl;
}
// END::TEST VECTORS LIST THREAD


//activeBlockTest001B();


  qPlayNext=true;
  //qPlayNext=false;

// BEGIN::TEST TG FROM SPECX
if (qPlayNext) {
    std::cout << std::endl;
    std::cout << "<<< Test Tg from Specx  >>>" << std::endl;
    activeBlock008();
    activeBlock008Beta();
    std::cout << std::endl;
}
// END::TEST TG FROM SPECX




  std::cout << "<<< The End >>>" << std::endl << std::endl;
  Color(7);
  return 0;
}
