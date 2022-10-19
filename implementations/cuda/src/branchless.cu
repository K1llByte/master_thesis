#include <iostream>
#include <functional>
#include <chrono>

#include <cufft.h>
#include "cuda_helper.hpp"

#define M_PI 3.1415926535897932384626433832795
#define FFT_SIZE 2048
#define LOG_SIZE 11
#define NUM_BUTTERFLIES 1
// #define BENCHMARK_RUNS 2

// Actual kernel functions

#define CU_ERR_CHECK_MSG(err, msg) {  \
    if(err != cudaSuccess) {          \
        fprintf(stderr, msg);         \
        exit(1);                      \
    }                                 \
}
#define CU_CHECK_MSG(res, msg) {  \
    if(res != CUFFT_SUCCESS) {    \
        fprintf(stderr, msg);     \
        exit(1);                  \
    }                             \
}

inline double benchmark(std::function<void()> func)
{
    double time_all = 0.;
    int NUM_RUNS = 5;
    for(int i = 0; i < NUM_RUNS; ++i) {
        auto begin = std::chrono::high_resolution_clock::now();
        func();
        auto end = std::chrono::high_resolution_clock::now();
        time_all += std::chrono::duration<double, std::milli>(end-begin).count();
    }
    return time_all / NUM_RUNS;
}


__device__ __forceinline__
float2 complex_mult(float2 v0, float2 v1) {
    return float2{
        v0.x * v1.x - v0.y * v1.y,
        v0.x * v1.y + v0.y * v1.x
    };
}

__device__ __forceinline__
float2 complex_add(float2 v0, float2 v1) {
    return float2{v0.x + v1.x, v0.y + v1.y };
}

__device__ __forceinline__
float2 complex_sub(float2 v0, float2 v1) {
    return float2{v0.x - v1.x, v0.y - v1.y };
}

__device__ __forceinline__
float2 euler(float angle) {
    return float2{cos(angle), sin(angle)};
}


// GPU Kernel
__global__
void stockham_fft_horizontal(float2* pingpong, float fft_dir) {
    int line = threadIdx.x;
    int column = blockIdx.x;
    int pingpong_idx = 0;

    for(int stage = 0; stage < LOG_SIZE; ++stage) {
        // 1. Compute Butterflies
        int n = 1 << (LOG_SIZE - stage);
        int m = n >> 1;
        int s = 1 << stage;

        int pp_read = pingpong_idx;
        int pp_write = (pingpong_idx+1) % 2;
        for(int i = 0; i < NUM_BUTTERFLIES; ++i) {
            int idx = (line*NUM_BUTTERFLIES + i);
            // Compute p and q
            int p = idx / s;
            int q = idx % s;
            // Compute the twiddle factor
            float2 wp = euler(fft_dir * 2 * (M_PI / n) * p);
            // Compute natural order butterflies
            float2 a = pingpong[pp_read*1 + (q + s*(p + 0))*2 + 2*FFT_SIZE*column];
            float2 b = pingpong[pp_read*1 + (q + s*(p + m))*2 + 2*FFT_SIZE*column];

            pingpong[pp_write*1 +  (q + s*(2*p + 0))*2 + 2*FFT_SIZE*column] = complex_add(a, b);
            pingpong[pp_write*1 +  (q + s*(2*p + 1))*2 + 2*FFT_SIZE*column] = complex_mult(wp,complex_sub(a, b));
        }

        // 2. Update Variables
        pingpong_idx = (pingpong_idx + 1) % 2;

        // 3. Sync by Memory Barrier
        __syncthreads();
    }
}


__global__
void stockham_fft_vertical(float2* pingpong, float fft_dir) {
    int line = blockIdx.x;
    int column = threadIdx.x;
    int pingpong_idx = LOG_SIZE % 2;

    for(int stage = 0; stage < LOG_SIZE; ++stage) {
        // 1. Compute Butterflies
        int n = 1 << (LOG_SIZE - stage);
        int m = n >> 1;
        int s = 1 << stage;

        int pp_read = pingpong_idx;
        int pp_write = (pingpong_idx+1) % 2;
        for(int i = 0; i < NUM_BUTTERFLIES; ++i) {
            int idx = (column*NUM_BUTTERFLIES + i);
            // Compute p and q
            int p = idx / s;
            int q = idx % s;
            // Compute the twiddle factor
            float2 wp = euler(fft_dir * 2 * (M_PI / n) * p);
            // Compute natural order butterflies
            float2 a = pingpong[pp_read + 2*(line + FFT_SIZE*(q + s*(p + 0)))];
            float2 b = pingpong[pp_read + 2*(line + FFT_SIZE*(q + s*(p + m)))];

            pingpong[pp_write + 2*line + 2*FFT_SIZE*(q + s*(2*p + 0))] = complex_add(a, b);
            pingpong[pp_write + 2*line + 2*FFT_SIZE*(q + s*(2*p + 1))] = complex_mult(wp,complex_sub(a, b));
        }

        // 2. Update Variables
        pingpong_idx = (pingpong_idx + 1) % 2;

        // 3. Sync by Memory Barrier
        __syncthreads();
    }
}


// Host code
int main() {
    cudaError_t err;
    ///////////////////////////////////////
    //     Comarison Implementation      //
    ///////////////////////////////////////
    const size_t CUFFT_BUFFER_SIZE_BYTES = FFT_SIZE*FFT_SIZE*sizeof(cufftComplex);
    cufftComplex* cufft_data = reinterpret_cast<cufftComplex*>(malloc(CUFFT_BUFFER_SIZE_BYTES));
    cufftComplex* gpu_data;

    // Initializing input sequence
    for(size_t i = 0; i < FFT_SIZE*FFT_SIZE; ++i) {
        cufft_data[i].x = i;
        cufft_data[i].y = 0.00;
    }

    // Allocate device memory
    err = cudaMalloc(&gpu_data, CUFFT_BUFFER_SIZE_BYTES);
    CU_ERR_CHECK_MSG(err, "Cuda error: Failed to allocate\n");

    // Copy data to GPU buffer
    err = cudaMemcpy(gpu_data, cufft_data, CUFFT_BUFFER_SIZE_BYTES, cudaMemcpyHostToDevice);
    CU_ERR_CHECK_MSG(err, "Cuda error: Failed to copy buffer to GPU\n");

    // Setup cufft plan
    cufftHandle plan;
    cufftResult_t res;
    res = cufftPlan2d(&plan, FFT_SIZE, FFT_SIZE, CUFFT_C2C);
    CU_CHECK_MSG(res, "cuFFT error: Plan creation failed\n");

    auto miliseconds = benchmark([&]() {
        // Submit execution
        res = cufftExecC2C(plan, gpu_data, gpu_data, CUFFT_FORWARD);
        CU_CHECK_MSG(res, "cuFFT error: ExecC2C Forward failed\n");
    
        // Await end of execution
        err = cudaDeviceSynchronize();
        CU_ERR_CHECK_MSG(err, "Cuda error: Failed to synchronize\n");
    });
    std::cout << "cuFFT: " << miliseconds << "ms\n";
    
    // Retrieve computed FFT buffer
    err = cudaMemcpy(cufft_data, gpu_data, CUFFT_BUFFER_SIZE_BYTES, cudaMemcpyDeviceToHost);
    CU_ERR_CHECK_MSG(err, "Cuda error: Failed to copy buffer to GPU\n");

    ///////////////////////////////////////
    //       CUDA Implementation         //
    ///////////////////////////////////////
    const size_t BUFFER_SIZE_BYTES = 2*FFT_SIZE*FFT_SIZE*sizeof(float2);

    // Allocate host memory
    float2* data;
    data = reinterpret_cast<float2*>(malloc(BUFFER_SIZE_BYTES));

    // Allocate device memory
    // Strided pingpong buffer
    float2* gpu_pingpong;
    err = cudaMalloc(&gpu_pingpong, BUFFER_SIZE_BYTES);
    CU_ERR_CHECK_MSG(err, "Cuda error: Failed to allocate\n");

    for(size_t i = 0; i < FFT_SIZE*FFT_SIZE; i++) {
        data[i*2].x = static_cast<float>(i);
        data[i*2].y = 0.f;
    }

    // Upload host data to device
    err = cudaMemcpy(gpu_pingpong, data, BUFFER_SIZE_BYTES, cudaMemcpyHostToDevice);
    CU_ERR_CHECK_MSG(err, "Cuda error: Failed to copy buffer to GPU\n");

    // std::cout << "Initialized Buffers on the GPU\n";
    // Compute FFT
    auto blocks = dim3(FFT_SIZE, 1);
    auto block_threads = dim3((FFT_SIZE / 2)/NUM_BUTTERFLIES, 1);


    std::cout << "CUDA Branchless: " << benchmark([&]() {
        // Horizontal pass
        // Sync after execution
        stockham_fft_horizontal<<<blocks, block_threads>>>(gpu_pingpong, -1.f);
        // err = cudaDeviceSynchronize();
        // CU_ERR_CHECK_MSG(err, "Cuda error: Failed to synchronize\n");

        // Vertical pass
        // Sync after execution
        stockham_fft_vertical<<<blocks, block_threads>>>(gpu_pingpong, -1.f);
        err = cudaDeviceSynchronize();
        CU_ERR_CHECK_MSG(err, "Cuda error: Failed to synchronize\n");
    }) << "ms\n";


    // Retrieve device data back to host
    err = cudaMemcpy(
        data,
        gpu_pingpong,
        BUFFER_SIZE_BYTES,
        cudaMemcpyDeviceToHost
    );
    CU_ERR_CHECK_MSG(err, "Cuda error: Failed to copy buffer from GPU\n");

    // // Print results
    // // CUDA
    // std::cout << "========= CUDA =========\n";
    // for(size_t i = 0; i < FFT_SIZE*FFT_SIZE; ++i) {
    //     std::cout << "(" << data[i].x << ", " << data[i].y << ")" << "\n";
    // }
    // // cuFFT
    // std::cout << "========= cuFFT =========\n";
    // for(size_t i = 0; i < FFT_SIZE*FFT_SIZE; ++i) {
    //     std::cout << "(" << cufft_data[i].x << ", " << cufft_data[i].y << ")" << "\n";
    // }

    // Free allocated resources
    cudaFree(gpu_pingpong);
    free(data);
}