#include <iostream>
#include <functional>
#include <chrono>

#include <cufft.h>
#include "cuda_helper.hpp"

#define M_PI 3.1415926535897932384626433832795
#define FFT_SIZE 1024
#define LOG_SIZE 10
#define HALF_LOG_SIZE 5
#define NUM_BUTTERFLIES 1
// #define BENCHMARK_RUNS 2

// Actual kernel functions

#define CU_ERR_CHECK_MSG(err, msg) {  \
    if(err != cudaSuccess) {          \
        fprintf(stderr, msg);         \
        fprintf(stderr, "%d", int(err));      \
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
    int NUM_RUNS = 10;
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
void stockham_fft_horizontal(float2* pingpong0, float2* pingpong1, float fft_dir) {
    int line = threadIdx.x;
    int column = blockIdx.x;
    int pingpong = 0;

    for(int stage = 0; stage < HALF_LOG_SIZE; ++stage) {
        // 1. Compute Butterflies

        int n = 1 << ((HALF_LOG_SIZE - stage)*2);
        int s = 1 << (stage*2);

        int n0 = 0;
        int n1 = n/4;
        int n2 = n/2;
        int n3 = n1 + n2; // 3N/4

        for(int i = 0; i < NUM_BUTTERFLIES; ++i) {
            int idx = (line*NUM_BUTTERFLIES + i);
            // Compute p and q
            int p = idx / s;
            int q = idx % s;
            // Compute the twiddle factors (2*(M_PI / n)) is constant
            float2 w1p = euler(-2*(M_PI / n) * p);
            float2 w2p = complex_mult(w1p,w1p);
            float2 w3p = complex_mult(w1p,w2p);
            if(pingpong == 0) {
                // Compute natural order butterflies
                float2 a = pingpong0[q + s*(p + n0) + FFT_SIZE*column];
                float2 b = pingpong0[q + s*(p + n1) + FFT_SIZE*column];
                float2 c = pingpong0[q + s*(p + n2) + FFT_SIZE*column];
                float2 d = pingpong0[q + s*(p + n3) + FFT_SIZE*column];

                float2 apc = complex_add(a,c);
                float2 amc = complex_sub(a,c);
                float2 bpd = complex_add(b,d);
                float2 jbmd = complex_mult(float2{0,1}, complex_sub(b,d));

                pingpong1[q + s*(4*p + 0) + FFT_SIZE*column] = complex_add(apc, bpd);
                pingpong1[q + s*(4*p + 1) + FFT_SIZE*column] = complex_mult(w1p, complex_sub(amc, jbmd));
                pingpong1[q + s*(4*p + 2) + FFT_SIZE*column] = complex_mult(w2p, complex_sub(apc, bpd));
                pingpong1[q + s*(4*p + 3) + FFT_SIZE*column] = complex_mult(w3p, complex_add(amc, jbmd));
            }
            else {
                // Compute natural order butterflies
                float2 a = pingpong1[q + s*(p + n0) + FFT_SIZE*column];
                float2 b = pingpong1[q + s*(p + n1) + FFT_SIZE*column];
                float2 c = pingpong1[q + s*(p + n2) + FFT_SIZE*column];
                float2 d = pingpong1[q + s*(p + n3) + FFT_SIZE*column];

                float2 apc = complex_add(a,c);
                float2 amc = complex_sub(a,c);
                float2 bpd = complex_add(b,d);
                float2 jbmd = complex_mult(float2{0,1}, complex_sub(b,d));

                pingpong0[q + s*(4*p + 0) + FFT_SIZE*column] = complex_add(apc, bpd);
                pingpong0[q + s*(4*p + 1) + FFT_SIZE*column] = complex_mult(w1p, complex_sub(amc, jbmd));
                pingpong0[q + s*(4*p + 2) + FFT_SIZE*column] = complex_mult(w2p, complex_sub(apc, bpd));
                pingpong0[q + s*(4*p + 3) + FFT_SIZE*column] = complex_mult(w3p, complex_add(amc, jbmd));
            }
        }

        // 2. Update Variables
        pingpong = ((pingpong + 1) % 2);

        // 3. Sync by Memory Barrier
        __syncthreads();
    }
}


__global__
void stockham_fft_vertical(float2* pingpong0, float2* pingpong1, float fft_dir) {
    int line = blockIdx.x;
    int column = threadIdx.x;
    int pingpong = HALF_LOG_SIZE % 2;

    for(int stage = 0; stage < HALF_LOG_SIZE; ++stage) {
        // 1. Compute Butterflies
        int n = 1 << ((HALF_LOG_SIZE - stage)*2);
        int s = 1 << (stage*2);

        int n0 = 0;
        int n1 = n/4;
        int n2 = n/2;
        int n3 = n1 + n2; // 3N/4

        for(int i = 0; i < NUM_BUTTERFLIES; ++i) {
            int idx = (column*NUM_BUTTERFLIES + i);
            // Compute p and q
            int p = idx / s;
            int q = idx % s;
            // Compute the twiddle factors (2*(M_PI / n)) is constant
            float2 w1p = euler(-2*(M_PI / n) * p);
            float2 w2p = complex_mult(w1p,w1p);
            float2 w3p = complex_mult(w1p,w2p);
            if(pingpong == 0) {
                // Compute natural order butterflies
                float2 a = pingpong0[line + FFT_SIZE*(q + s*(p + n0))];
                float2 b = pingpong0[line + FFT_SIZE*(q + s*(p + n1))];
                float2 c = pingpong0[line + FFT_SIZE*(q + s*(p + n2))];
                float2 d = pingpong0[line + FFT_SIZE*(q + s*(p + n3))];

                float2 apc = complex_add(a,c);
                float2 amc = complex_sub(a,c);
                float2 bpd = complex_add(b,d);
                float2 jbmd = complex_mult(float2{0,1}, complex_sub(b,d));

                pingpong1[line + FFT_SIZE*(q + s*(4*p + 0))] = complex_add(apc, bpd);
                pingpong1[line + FFT_SIZE*(q + s*(4*p + 1))] = complex_mult(w1p, complex_sub(amc, jbmd));
                pingpong1[line + FFT_SIZE*(q + s*(4*p + 2))] = complex_mult(w2p, complex_sub(apc, bpd));
                pingpong1[line + FFT_SIZE*(q + s*(4*p + 3))] = complex_mult(w3p, complex_add(amc, jbmd));
            }
            else {
                // Compute natural order butterflies
                float2 a = pingpong1[line + FFT_SIZE*(q + s*(p + n0))];
                float2 b = pingpong1[line + FFT_SIZE*(q + s*(p + n1))];
                float2 c = pingpong1[line + FFT_SIZE*(q + s*(p + n2))];
                float2 d = pingpong1[line + FFT_SIZE*(q + s*(p + n3))];

                float2 apc = complex_add(a,c);
                float2 amc = complex_sub(a,c);
                float2 bpd = complex_add(b,d);
                float2 jbmd = complex_mult(float2{0,1}, complex_sub(b,d));

                pingpong0[line + FFT_SIZE*(q + s*(4*p + 0))] = complex_add(apc, bpd);
                pingpong0[line + FFT_SIZE*(q + s*(4*p + 1))] = complex_mult(w1p, complex_sub(amc, jbmd));
                pingpong0[line + FFT_SIZE*(q + s*(4*p + 2))] = complex_mult(w2p, complex_sub(apc, bpd));
                pingpong0[line + FFT_SIZE*(q + s*(4*p + 3))] = complex_mult(w3p, complex_add(amc, jbmd));
            }
        }

        // 2. Update Variables
        pingpong = (pingpong + 1) % 2;

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
    const size_t BUFFER_SIZE_BYTES = FFT_SIZE*FFT_SIZE*sizeof(float2);

    // Allocate host memory
    float2* data;
    data = reinterpret_cast<float2*>(malloc(BUFFER_SIZE_BYTES));
    
    // Allocate device memory
    float2* gpu_pingpong0;
    float2* gpu_pingpong1;
    err = cudaMalloc(&gpu_pingpong0, BUFFER_SIZE_BYTES);
    CU_ERR_CHECK_MSG(err, "Cuda error: Failed to allocate\n");
    err = cudaMalloc(&gpu_pingpong1, BUFFER_SIZE_BYTES);
    CU_ERR_CHECK_MSG(err, "Cuda error: Failed to allocate\n");

    for(size_t i = 0; i < FFT_SIZE*FFT_SIZE; i++) {
        data[i].x = static_cast<float>(i);
        data[i].y = 0.f;
    }

    // Upload host data to device
    err = cudaMemcpy(gpu_pingpong0, data, BUFFER_SIZE_BYTES, cudaMemcpyHostToDevice);
    CU_ERR_CHECK_MSG(err, "Cuda error: Failed to copy buffer to GPU\n");

    // std::cout << "Initialized Buffers on the GPU\n";
    // Compute FFT
    auto blocks = dim3(FFT_SIZE, 1);
    auto block_threads = dim3((FFT_SIZE / 4)/NUM_BUTTERFLIES, 1);

    // std::cout << "CUDA Horizontal: " << benchmark([&]() {
    //     // Horizontal pass
    //     // Sync after execution
    //     stockham_fft_horizontal<<<blocks, block_threads>>>(gpu_pingpong0, gpu_pingpong1, -1.f);
    //     err = cudaDeviceSynchronize();
    //     CU_ERR_CHECK_MSG(err, "Cuda error: Failed to synchronize\n");
    //     // std::cout << "Sinchronized after execution\n";
    // }) << "ms\n";

    // std::cout << "CUDA Vertical: " << benchmark([&]() {
    //     // Vertical pass
    //     // Sync after execution
    //     stockham_fft_vertical<<<blocks, block_threads>>>(gpu_pingpong0, gpu_pingpong1, -1.f);
    //     err = cudaDeviceSynchronize();
    //     CU_ERR_CHECK_MSG(err, "Cuda error: Failed to synchronize\n");
    //     // std::cout << "Sinchronized after execution\n";
    // }) << "ms\n";

    std::cout << "CUDA: " << benchmark([&]() {
        // Horizontal pass
        // Sync after execution
        stockham_fft_horizontal<<<blocks, block_threads>>>(gpu_pingpong0, gpu_pingpong1, -1.f);
        // err = cudaDeviceSynchronize();
        // CU_ERR_CHECK_MSG(err, "Cuda error: Failed to synchronize\n");

        // Vertical pass
        // Sync after execution
        stockham_fft_vertical<<<blocks, block_threads>>>(gpu_pingpong0, gpu_pingpong1, -1.f);
        err = cudaDeviceSynchronize();
        CU_ERR_CHECK_MSG(err, "Cuda error: Failed to synchronize\n");
    }) << "ms\n";


    // Retrieve device data back to host
    err = cudaMemcpy(
        data,
        (LOG_SIZE % 2 == 0) ? gpu_pingpong0 : gpu_pingpong1,
        BUFFER_SIZE_BYTES,
        cudaMemcpyDeviceToHost
    );
    CU_ERR_CHECK_MSG(err, "Cuda error: Failed to copy buffer from GPU\n");

    // Print results
    // CUDA
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
    cudaFree(gpu_pingpong0);
    cudaFree(gpu_pingpong1);
    free(data);
}