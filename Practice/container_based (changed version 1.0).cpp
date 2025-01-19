#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExecuteForEachAccTag.hpp>

#include <array>
#include <iostream>
#include <type_traits>

#include <boost/numeric/odeint.hpp>

#include <boost/timer.hpp>
#include <boost/random/cauchy_distribution.hpp>
using namespace boost::numeric::odeint;

namespace CoolFunctions{

    template<typename T>
    concept IsVec = requires(T v) {
        typename T::DataType;
        typename T::TAcc;
    };

    template<typename TAccel, typename T, typename contT1,  typename contT2>
    struct Add{
        contT1 op1;
        contT2 op2;
        using DataType = T;
        using TAcc = TAccel;
        ALPAKA_FN_ACC DataType Get(TAcc const& acc, int i){
            return op1.Get(acc, i) + op2.Get(acc, i);
        }
    };

    template<typename TAcc, typename T, typename contT1, typename contT2>
    ALPAKA_FN_INLINE ALPAKA_FN_ACC
    auto make_add(contT1&& op1, contT2&& op2) {
        return Add<TAcc, T, std::decay_t<contT1>, std::decay_t<contT2>>{
            std::forward<contT1>(op1), std::forward<contT2>(op2)};
    }

    template<typename TAccel, typename T>
    requires(std::is_arithmetic_v<T>)
    struct CoolNumber{
        using DataType = T;
        using TAcc = TAccel;
        using DevAcc = alpaka::Dev<TAcc>; 
        DataType val;
        ALPAKA_FN_ACC DataType Get(TAcc const& acc, int i){
            return val;
        }
    };
    

    template<typename TAccel, typename T, typename QType>
    struct  CoolVec{
        using DataType = T;
        using TAcc = TAccel;
        using DevAcc = alpaka::Dev<TAcc>;
        
        using Idx = std::size_t;
        using Dim = alpaka::DimInt<1>;
        using BufAcc = alpaka::Buf<DevAcc, DataType, Dim, Idx>;
        using DevHost = alpaka::DevCpu;

        DataType* ptr;
        QType* queue;
        DevAcc* acc;
        Idx id;
        size_t size;
        ALPAKA_FN_INLINE ALPAKA_FN_ACC DataType Get(TAcc const& acc, int i) const {
            return ptr[i];
        }
        ALPAKA_FN_INLINE ALPAKA_FN_ACC void Set(TAcc const& acc, T val, int i)  {
            ptr[i] = val;
        }


        // template<typename TAccel, typename T, typename QType, typename Operation>
        template<typename Operation>
        CoolVec& operator=(const Operation& operation) {
            static_assert(IsVec<Operation>, "RHS must be a valid vector operation!");
            static_assert(
                std::is_same_v<typename Operation::DataType, DataType>,
                "Data types of LHS and RHS must match!");
            static_assert(
                std::is_same_v<typename Operation::TAcc, TAcc>,
                "Accelerator types of LHS and RHS must match!");
            execute_kernel(*this, operation);

            // Perform the kernel execution
            return *this; // Return *this to allow chaining
        }

        CoolVec(DataType* ptr,  QType* queue, DevAcc* acc, size_t size):ptr(ptr), queue(queue), acc(acc), size(size), id(0u){}
        CoolVec(DataType* ptr,  QType* queue, DevAcc* acc, size_t size, Idx id):ptr(ptr), queue(queue), acc(acc), size(size), id(id){}

        CoolVec(  QType* queue, DevAcc* acc): queue(queue), acc(acc){}

        // Defaulted special member functions for triviality
        CoolVec() = default;
        CoolVec(const CoolVec&) = default;
        CoolVec& operator=(const CoolVec&) = default;
        CoolVec(CoolVec&&) = default;
        CoolVec& operator=(CoolVec&&) = default;
        ~CoolVec() = default;
    };
    template<typename TAccel, typename T, typename QType>
    struct  CoolContainer{
        using DataType = T;
        using TAcc = TAccel;
        using DevAcc = alpaka::Dev<TAcc>;
        
        using Idx = std::size_t;
        using Dim = alpaka::DimInt<1>;
        using BufAcc = alpaka::Buf<DevAcc, DataType, Dim, Idx>;
        using DevHost = alpaka::DevCpu;
        using BufHost = alpaka::Buf<DevHost, DataType, Dim, Idx>;
        static BufHost makeHostBuf(std::size_t size)
        {
            auto const devHost = alpaka::getDevByIdx(alpaka::PlatformCpu{}, 0);
            alpaka::Vec<Dim, Idx> extent(size);
            return alpaka::allocBuf<T, Idx>(devHost, extent);
        }

        // Helper: allocate on device
        static BufAcc makeDevBuf(DevAcc const& dev, std::size_t size)
        {
            alpaka::Vec<Dim, Idx> extent(size);
            return alpaka::allocBuf<T, Idx>(dev, extent);
        }

            CoolContainer() = default;
            CoolContainer(  QType* queue, DevAcc* acc, std::size_t size): queue(queue), acc(acc), host_buffer(makeHostBuf(size)), buffer(makeDevBuf(*acc, size)), size(size){
                MakeConstVec();
            }
            CoolContainer(  QType* queue, DevAcc* acc, std::size_t size, DataType value): queue(queue), acc(acc), host_buffer(makeHostBuf(size)), buffer(makeDevBuf(*acc, size)), size(size){
                MakeConstVec(value);
            }

           void MakeConstVec(DataType value = 0.0){
                for(int i = 0; i < size; ++i){
                    host_buffer[i] = value;
                }
                alpaka::memcpy(*queue, buffer, host_buffer);
                alpaka::wait(*queue);
                vector =  CoolVec<TAcc, DataType, QType>{std::data(buffer), queue, acc, size, current_id++}; 

            }


            void  MakeFromContainer(std::vector<DataType> data){
  
                for(int i = 0; i < size; ++i){
                    host_buffer[i] = data[i];
                }
                alpaka::memcpy(*queue, buffer, host_buffer);
                alpaka::wait(*queue);
                vector = CoolVec<TAcc, DataType, QType>{std::data(buffer), queue, acc, size, current_id}; 
            }

            std::vector<DataType> GetVector(){
                alpaka::memcpy(*queue, host_buffer, buffer);
                alpaka::wait(*queue);
                std::vector<DataType> hostData(size);
                for (Idx i = 0; i < size; ++i) {
                    hostData[i] = host_buffer[i];
                }
                return hostData;
            }

            
        CoolVec<TAcc, DataType, QType> vector;
        template<typename Operation>
        CoolContainer& operator=(const Operation& operation) {
            this->vector = operation;
            return *this; // Return *this to allow chaining
        }


        private:
            QType* queue;
            DevAcc* acc;
            BufAcc buffer;
            BufHost host_buffer;
            Idx current_id;
            std::size_t size;

    };
    
    class ExecuteExpressionKernel
    {
    public:

        ALPAKA_NO_HOST_ACC_WARNING
        template<typename TAcc, typename TElem, typename RElem, typename TIdx>
        ALPAKA_FN_ACC auto operator()(
            TAcc const& acc,
            RElem  result, 
            TElem operation,
            TIdx const& numElements) const -> void
        {
            static_assert(alpaka::Dim<TAcc>::value == 1, "The ExecuteExpressionKernel expects 1-dimensional indices!");

            // The uniformElements range for loop takes care automatically of the blocks, threads and elements in the
            // kernel launch grid.
            for(auto i : alpaka::uniformElements(acc, numElements))
            {
                
                result.Set(acc,operation.Get(acc, i), i);
            }
        } 
    };

    class HandWrittenKernel
    {
    public:
        ALPAKA_NO_HOST_ACC_WARNING
	template<typename TAcc, typename TElem, typename TIdx>
	ALPAKA_FN_ACC auto operator()
	(
		TAcc const& acc,
		TElem const* const A,
		TElem const* const B,
		TElem const* const C,
		TElem const* const D,
		TElem const* const E,
		TElem* const result,
		TIdx const& numElements) const -> void
	{
		for (auto i : alpaka::uniformElements(acc, numElements))
		{
			result[i] = B[i] + A[i] + C[i] + D[i] + E[i] + 1;
		}
	}
    };




    // Overloaded operator= to trigger the kernel
    template<typename TAccel, typename T, typename QType, typename Operation>
    void execute_kernel(CoolVec<TAccel, T, QType>& result, const Operation& operation) {
        std::size_t numElements = result.size; // Example size, this should be parameterized
        ExecuteExpressionKernel kernel;

        alpaka::KernelCfg<TAccel> const kernelCfg = {numElements, 1u};
        auto const workDiv = alpaka::getValidWorkDiv(
            kernelCfg,
            *result.acc, // Alpaka device
            kernel,
            result,
            operation,
            numElements);

        auto const taskKernel = alpaka::createTaskKernel<TAccel>(
            workDiv,
            kernel,
            result,
            operation,
            numElements);

        alpaka::enqueue(*result.queue, taskKernel);
        alpaka::wait(*result.queue); // Wait for kernel execution to complete
    }


    template<typename Lhs, typename Rhs>
    requires requires(Lhs lhs, Rhs rhs) {
        lhs.vector; // Check if Lhs has Get
        rhs.vector; // Check if Lhs has Get
    } && std::is_same_v<typename std::decay_t<Lhs>::DataType, typename std::decay_t<Rhs>::DataType> &&
            std::is_same_v<typename std::decay_t<Lhs>::TAcc, typename std::decay_t<Rhs>::TAcc>
    auto operator+(Lhs&& lhs, Rhs&& rhs){
        return lhs.vector + rhs.vector;
    }

    template<typename Lhs, typename Rhs>
    requires requires(Lhs lhs, Rhs rhs) {
        lhs.vector; // Check if Lhs has Get
        rhs.Get(std::declval<typename std::decay_t<Rhs>::TAcc>(), 0); // Check if Rhs has Get
    } && std::is_same_v<typename std::decay_t<Lhs>::DataType, typename std::decay_t<Rhs>::DataType> &&
            std::is_same_v<typename std::decay_t<Lhs>::TAcc, typename std::decay_t<Rhs>::TAcc>
    auto operator+(Lhs&& lhs, Rhs&& rhs){
        return lhs.vector + rhs;
    }

    template<typename Lhs, typename Rhs>
    requires requires(Lhs lhs, Rhs rhs) {
        lhs.Get(std::declval<typename std::decay_t<Lhs>::TAcc>(), 0); // Check if Lhs has Get
        rhs.vector; // Check if Lhs has Get
    } && std::is_same_v<typename std::decay_t<Lhs>::DataType, typename std::decay_t<Rhs>::DataType> &&
            std::is_same_v<typename std::decay_t<Lhs>::TAcc, typename std::decay_t<Rhs>::TAcc>
    auto operator+(Lhs&& lhs, Rhs&& rhs){
        return lhs + rhs.vector;
    }


    template<typename Lhs, typename Rhs>
    requires requires(Lhs lhs, Rhs rhs) {
        lhs.Get(std::declval<typename std::decay_t<Lhs>::TAcc>(), 0); // Check if Lhs has Get
        rhs.Get(std::declval<typename std::decay_t<Lhs>::TAcc>(), 0); // Check if Lhs has Get

    } && std::is_same_v<typename std::decay_t<Lhs>::DataType, typename std::decay_t<Rhs>::DataType> &&
            std::is_same_v<typename std::decay_t<Lhs>::TAcc, typename std::decay_t<Rhs>::TAcc>
    auto operator+(Lhs&& lhs, Rhs&& rhs) {
        using TAcc = typename std::decay_t<Lhs>::TAcc;
        using DataType = typename std::decay_t<Lhs>::DataType;

        return make_add<TAcc, DataType>(std::forward<Lhs>(lhs), std::forward<Rhs>(rhs));
    }

    template<typename Lhs, typename Rhs>
    requires (std::is_arithmetic_v<std::decay_t<Lhs>> && 
            requires(Rhs rhs) {
                rhs.Get(std::declval<typename std::decay_t<Rhs>::TAcc>(), 0); // Rhs has Get
            } && 
            !requires(Lhs lhs, Rhs rhs) { // Ensure this is not matched by the first operator
                lhs.Get(std::declval<typename std::decay_t<Lhs>::TAcc>(), 0);
                rhs.Get(std::declval<typename std::decay_t<Rhs>::TAcc>(), 0);
            })
    auto operator+(Lhs&& lhs, Rhs&& rhs) {
        using TAcc = typename std::decay_t<Rhs>::TAcc;
        using DataType = typename std::decay_t<Rhs>::DataType;
        
        auto const number = CoolNumber<TAcc, DataType>(std::forward<Lhs>(lhs));
        return make_add<TAcc, DataType>(number, std::forward<Rhs>(rhs));
    }


    template<typename Lhs, typename Rhs>
    requires (std::is_arithmetic_v<std::decay_t<Rhs>> && 
            requires(Lhs lhs) {
                lhs.Get(std::declval<typename std::decay_t<Lhs>::TAcc>(), 0); // Rhs has Get
            } && 
            !requires(Lhs lhs, Rhs rhs) { // Ensure this is not matched by the first operator
                lhs.Get(std::declval<typename std::decay_t<Lhs>::TAcc>(), 0);
                rhs.Get(std::declval<typename std::decay_t<Rhs>::TAcc>(), 0);
            })
    auto operator+(Lhs&& lhs, Rhs&& rhs) {
        using TAcc = typename std::decay_t<Lhs>::TAcc;
        using DataType = typename std::decay_t<Lhs>::DataType;
        
        auto const number = CoolNumber<TAcc, DataType>(std::forward<Rhs>(rhs));
        return make_add<TAcc, DataType>(std::forward<Lhs>(lhs),number );
    }

}




template<typename TAcc>
void example(){
    using namespace CoolFunctions;
    using Dim = alpaka::DimInt<1u>;
    using Idx = std::size_t;

    // Define the accelerator
    using Acc = alpaka::TagToAcc<TAcc, Dim, Idx>;
    using DevAcc = alpaka::Dev<Acc>;
    std::cout << "Using alpaka accelerator: " << alpaka::getAccName<Acc>() << std::endl;

    // Defines the synchronization behavior of a queue
    //
    // choose between Blocking and NonBlocking
    using QueueProperty = alpaka::Blocking;
    using QueueAcc = alpaka::Queue<Acc, QueueProperty>;

    // Select a device
    auto const platform = alpaka::Platform<Acc>{};
    auto  devAcc = alpaka::getDevByIdx(platform, 0);

    // Create a queue on thea device
    QueueAcc queue(devAcc);

    // Define the work division
    //Idx const numElements(120);
    Idx const numElements(123456);
    Idx const elementsPerThread(8u);
    alpaka::Vec<Dim, Idx> const extent(numElements);
    alpaka::Vec<Dim, Idx> const extent_1(1);

    // Define the buffer element type
    using Data = double;

    // Get the host device for allocating memory on the host.
    using DevHost = alpaka::DevCpu;
    auto const platformHost = alpaka::PlatformCpu{};
    auto const devHost = alpaka::getDevByIdx(platformHost, 0);

    // Allocate 3 host memory buffers
    using BufHost = alpaka::Buf<DevHost, Data, Dim, Idx>;
    BufHost buffHostA(alpaka::allocBuf<Data, Idx>(devHost, extent));
    BufHost buffHostB(alpaka::allocBuf<Data, Idx>(devHost, extent));
    BufHost buffHostC(alpaka::allocBuf<Data, Idx>(devHost, extent));
    BufHost buffHostD(alpaka::allocBuf<Data, Idx>(devHost, extent));
    BufHost buffHostE(alpaka::allocBuf<Data, Idx>(devHost, extent));
    BufHost buffHostResult(alpaka::allocBuf<Data, Idx>(devHost, extent));

    std::random_device rd{};
    std::default_random_engine eng{rd()};
    std::uniform_real_distribution<Data> dist(1.0, 42.0);

    for (Idx i(0); i < numElements; ++i)
    {
	    buffHostA[i] = dist(eng);
	    buffHostB[i] = dist(eng);
	    buffHostC[i] = dist(eng);
	    buffHostD[i] = dist(eng);
	    buffHostE[i] = dist(eng);
	    
	    buffHostResult[i] = 0;
    }

    using BufAcc = alpaka::Buf<DevAcc, Data, Dim, Idx>;
    BufAcc buffAccA(alpaka::allocBuf<Data, Idx>(devAcc, extent));
    BufAcc buffAccB(alpaka::allocBuf<Data, Idx>(devAcc, extent));
    BufAcc buffAccC(alpaka::allocBuf<Data, Idx>(devAcc, extent));
    BufAcc buffAccD(alpaka::allocBuf<Data, Idx>(devAcc, extent));
    BufAcc buffAccE(alpaka::allocBuf<Data, Idx>(devAcc, extent));
    BufAcc buffAccResult(alpaka::allocBuf<Data, Idx>(devAcc, extent));

    alpaka::memcpy(queue, buffAccA, buffHostA);
    alpaka::memcpy(queue, buffAccB, buffHostB);
    alpaka::memcpy(queue, buffAccC, buffHostC);
    alpaka::memcpy(queue, buffAccD, buffHostD);
    alpaka::memcpy(queue, buffAccE, buffHostE);
    alpaka::memcpy(queue, buffAccResult, buffHostResult);

    // HandWritten Kernel
    HandWrittenKernel kernel;

    alpaka::KernelCfg<Acc> const kernelCfg = {extent, elementsPerThread};

    auto const workDiv = alpaka::getValidWorkDiv(
        kernelCfg,
        devAcc,
        kernel,
        alpaka::getPtrNative(buffAccA),
        alpaka::getPtrNative(buffAccB),
        alpaka::getPtrNative(buffAccC),
        alpaka::getPtrNative(buffAccD),
        alpaka::getPtrNative(buffAccE),
        alpaka::getPtrNative(buffAccResult),
        numElements);

    // Create the kernel execution task.
    auto const taskKernel = alpaka::createTaskKernel<Acc>(
        workDiv,
        kernel,
        std::data(buffAccA),
        std::data(buffAccB),
        std::data(buffAccC),
        std::data(buffAccD),
        std::data(buffAccE),
        std::data(buffAccResult),
        numElements);


    CoolContainer<Acc, Data, QueueAcc> testContainerA(&queue, &devAcc, numElements,5);
    CoolContainer<Acc, Data, QueueAcc> testContainerB(&queue, &devAcc, numElements,10);
    CoolContainer<Acc, Data, QueueAcc> testContainerC(&queue, &devAcc, numElements);
    CoolContainer<Acc, Data, QueueAcc> testContainerD(&queue, &devAcc, numElements);
    CoolContainer<Acc, Data, QueueAcc> testContainerE(&queue, &devAcc, numElements);

    CoolContainer<Acc, Data, QueueAcc> result(&queue, &devAcc, numElements);

    std::vector<double> times;
    //std::ofstream resultsFile("./benchmark_results.csv", std::ios::app);


    {
	alpaka::wait(queue);
        auto const beginT = std::chrono::high_resolution_clock::now();
        alpaka::enqueue(queue, taskKernel);
        // wait in case we are using an asynchronous queue to time actual kernel runtime
        alpaka::wait(queue);
        auto const endT = std::chrono::high_resolution_clock::now();
        std::cout << "Time for Custom kernel execution: " << std::chrono::duration<double>(endT - beginT).count() << 's'
                  << std::endl;
    }

    {
        auto beginT = std::chrono::high_resolution_clock::now();
        alpaka::memcpy(queue, buffHostResult, buffAccResult);
        alpaka::wait(queue);
        auto const endT = std::chrono::high_resolution_clock::now();
        //std::cout << "Time for HtoD copy: " << std::chrono::duration<double>(endT - beginT).count() << 's'
          //        << std::endl;
    }

{   

    for (int i = 0; i < 10; ++i) {
	auto const beginT = std::chrono::high_resolution_clock::now();

        result =  testContainerB + testContainerA +  testContainerC + testContainerD + testContainerE+1; // Run the kernel


        auto const endT = std::chrono::high_resolution_clock::now();

        //auto result_vec = result.GetVector();

        //for(Data v: result_vec){
        //std::cout << v << std::endl;	
        //}
        //std::cout << "Time for kernel executionkkk: " << std::chrono::duration<double>(endT - beginT).count() << 's'
          //        << std::endl;
	  
	double duration = std::chrono::duration<double>(endT - beginT).count();

//	resultsFile << "Run " << i << " : " << duration << "s" << std::endl;
	times.push_back(duration);
        }
    
    double avg = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    double min_time = *std::min_element(times.begin(), times.end());
    double max_time = *std::max_element(times.begin(), times.end());

    std::cout << "Average time : " << avg << "s" << std::endl;
    //std::cout << "Min time : " << min_time << "s" << std::endl;
    //std::cout << "Max time : " << max_time << "s" << std::endl;
    std::cout << std::endl;


}
}


int main(){
     std::cout << "Check enabled accelerator tags:" << std::endl;
    example<alpaka::TagGpuCudaRt>();
    example<alpaka::TagCpuSerial>();
    // thrustExample<alpaka::TagGpuCudaRt>();
    // thrustExample<alpaka::TagCpuSerial>();
    return 0;
}
