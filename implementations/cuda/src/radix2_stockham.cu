#include <iostream>
#include <functional>
#include <chrono>

#include <cufft.h>
#include "cuda_helper.hpp"

#define M_PI 3.1415926535897932384626433832795
#define FFT_SIZE 1024
#define LOG_SIZE 10
#define HALF_LOG_SIZE 4
#define NUM_BUTTERFLIES 1
#define BENCHMARK_RUNS 30

// Actual kernel functions

#define CU_ERR_CHECK_MSG(err, msg) {  \
    if(err != cudaSuccess) {          \
        fprintf(stderr, msg);         \
        fprintf(stderr, "%d", int(err)); \
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
    // TODO_ Skiup first iter
    double time_all = 0.;
    for(int i = 0; i < BENCHMARK_RUNS; ++i) {
        auto begin = std::chrono::high_resolution_clock::now();
        func();
        auto end = std::chrono::high_resolution_clock::now();
        time_all += std::chrono::duration<double, std::milli>(end-begin).count();
    }
    return time_all / BENCHMARK_RUNS;
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


__global__
void stockham_fft_horizontal(cudaSurfaceObject_t pingpong0, cudaSurfaceObject_t pingpong1, float fft_dir) {
    int line = threadIdx.x;
    int column = blockIdx.x;
    int pingpong = 0;

    for(int stage = 0; stage < LOG_SIZE; ++stage) {
        // 1. Compute Butterflies
        int n = 1 << (LOG_SIZE - stage);
        int m = n >> 1;
        int s = 1 << stage;
        for(int i = 0; i < NUM_BUTTERFLIES; ++i) {
            int idx = (line*NUM_BUTTERFLIES + i);
            // Compute p and q
            int p = idx / s;
            int q = idx % s;
            // Compute the twiddle factor
            float2 wp = euler(fft_dir * 2 * (M_PI / n) * p);
            if(pingpong == 0) {
                // Compute natural order butterflies
                float2 a = surf2Dread<float2>(pingpong0, (q + s*(p + 0))*sizeof(float2), column);
                float2 b = surf2Dread<float2>(pingpong0, (q + s*(p + m))*sizeof(float2), column);

                surf2Dwrite<float2>(complex_add(a, b),                  pingpong1, (q + s*(2*p + 0))*sizeof(float2), column);
                surf2Dwrite<float2>(complex_mult(wp,complex_sub(a, b)), pingpong1, (q + s*(2*p + 1))*sizeof(float2), column);
            }
            else {
                // Compute natural order butterflies
                float2 a = surf2Dread<float2>(pingpong1, (q + s*(p + 0))*sizeof(float2), column);
                float2 b = surf2Dread<float2>(pingpong1, (q + s*(p + m))*sizeof(float2), column);

                surf2Dwrite<float2>(complex_add(a, b),                  pingpong0, (q + s*(2*p + 0))*sizeof(float2), column);
                surf2Dwrite<float2>(complex_mult(wp,complex_sub(a, b)), pingpong0, (q + s*(2*p + 1))*sizeof(float2), column);
            }
        }

        // 2. Update Variables
        pingpong = ((pingpong + 1) % 2);

        // 3. Sync by Memory Barrier
        __syncthreads();
    }
}


__global__
void stockham_fft_vertical(cudaSurfaceObject_t pingpong0, cudaSurfaceObject_t pingpong1, float fft_dir) {
    int line = blockIdx.x;
    int column = threadIdx.x;
    int pingpong = LOG_SIZE % 2;

    for(int stage = 0; stage < LOG_SIZE; ++stage) {
        // 1. Compute Butterflies
        int n = 1 << (LOG_SIZE - stage);
        int m = n >> 1;
        int s = 1 << stage;

        float mult_factor = 1.f;
	    if ((stage == LOG_SIZE - 1) && fft_dir == 1) {
	    	mult_factor = 1.f / (FFT_SIZE*FFT_SIZE);
	    }

        for(int i = 0; i < NUM_BUTTERFLIES; ++i) {
            int idx = (column*NUM_BUTTERFLIES + i);
            // Compute p and q
            int p = idx / s;
            int q = idx % s;
            // Compute the twiddle factor
            float2 wp = euler(fft_dir * 2 * (M_PI / n) * p);
            if(pingpong == 0) {
                // Compute natural order butterflies
                float2 a = surf2Dread<float2>(pingpong0, line*sizeof(float2), (q + s*(p + 0)));
                float2 b = surf2Dread<float2>(pingpong0, line*sizeof(float2), (q + s*(p + m)));

                float2 ra = complex_add(a, b);
                float2 rb = complex_mult(wp,complex_sub(a, b));
                
                surf2Dwrite<float2>(ra * mult_factor, pingpong1, line*sizeof(float2), (q + s*(2*p + 0)));
                surf2Dwrite<float2>(rb * mult_factor, pingpong1, line*sizeof(float2), (q + s*(2*p + 1)));
            }
            else {
                // Compute natural order butterflies
                float2 a = surf2Dread<float2>(pingpong0, line*sizeof(float2), (q + s*(p + 0)));
                float2 b = surf2Dread<float2>(pingpong0, line*sizeof(float2), (q + s*(p + m)));

                float2 ra = complex_add(a, b);
                float2 rb = complex_mult(wp,complex_sub(a, b));
                
                surf2Dwrite<float2>(ra * mult_factor, pingpong1, line*sizeof(float2), (q + s*(2*p + 0)));
                surf2Dwrite<float2>(rb * mult_factor, pingpong1, line*sizeof(float2), (q + s*(2*p + 1)));
            }
        }

        // 2. Update Variables
        pingpong = (pingpong + 1) % 2;

        // 3. Sync by Memory Barrier
        __syncthreads();
    }
}


struct Texture {
    cudaSurfaceObject_t surface = 0;
    cudaArray_t array;
};
Texture create_surface(float* source = nullptr) {
    Texture res;
    // Allocate CUDA arrays in device memory
    cudaChannelFormatDesc channelDesc =
        cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindFloat);
    cudaMallocArray(&res.array, &channelDesc, FFT_SIZE, FFT_SIZE,
                    cudaArraySurfaceLoadStore);

    if(source != nullptr) {
        // Set pitch of the source (the width in memory in bytes of the 2D array
        // pointed to by src, including padding), we dont have any padding
        const size_t spitch = FFT_SIZE * 2*sizeof(float);
        // Copy data located at address source in host memory to device memory
        cudaMemcpy2DToArray(res.array, 0, 0, source, spitch,
                            FFT_SIZE * 2*sizeof(float), FFT_SIZE,
                            cudaMemcpyHostToDevice);
    }

    // Specify surface
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = res.array;
    
    // Create the surface objects
    cudaCreateSurfaceObject(&res.surface, &resDesc);
    return res;
}

void destroy_surface(Texture& tex) {
    cudaDestroySurfaceObject(tex.surface);
    cudaFreeArray(tex.array);
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
    const size_t BUFFER_SIZE_BYTES = FFT_SIZE*FFT_SIZE*sizeof(float)*2;

    // Allocate host memory
    float* data;
    data = reinterpret_cast<float*>(malloc(BUFFER_SIZE_BYTES));
    // Initialize host memory
    for(size_t i = 0; i < FFT_SIZE*FFT_SIZE; i++) {
        data[i*2] = static_cast<float>(i);
        data[(i+1)*2] = 0.f;
    }
    
    // Allocate device memory
    auto gpu_pp0 = create_surface(data);
    auto gpu_pp1 = create_surface();

    // Compute FFT
    auto blocks = dim3(FFT_SIZE, 1);
    auto block_threads = dim3((FFT_SIZE / 2)/NUM_BUTTERFLIES, 1);

    // Init Benchmark stuff
    cudaEvent_t start_event;
    cudaEvent_t end_event;
    CU_ERR_CHECK_MSG(cudaEventCreate(&start_event), "");
    CU_ERR_CHECK_MSG(cudaEventCreate(&end_event), "");

    float gpu_time_sum;
    size_t bench_run = 0;
    std::cout << "CUDA CPU: " << benchmark([&]() {
        // Record event before execution on stream 0 (default)
        CU_ERR_CHECK_MSG(cudaEventRecord(start_event, 0), "");

        // Horizontal pass
        stockham_fft_horizontal<<<blocks, block_threads>>>(gpu_pp0.surface, gpu_pp1.surface, -1.f);

        // Vertical pass
        stockham_fft_vertical<<<blocks, block_threads>>>(gpu_pp0.surface, gpu_pp1.surface, -1.f);

        // Record event after execution on stream 0 (default)
        CU_ERR_CHECK_MSG(cudaEventRecord(end_event, 0), "");
        
        // Sync after execution
        err = cudaDeviceSynchronize();
        // CU_ERR_CHECK_MSG(err, "Cuda error: Failed to synchronize\n");
        // CU_ERR_CHECK_MSG(cudaGetLastError(), "Cuda error: Failed to synchronize\n")

        
        if(bench_run != 0) {
            float gpu_time;
            CU_ERR_CHECK_MSG(cudaEventElapsedTime(&gpu_time, start_event, end_event), "");
            gpu_time_sum += gpu_time;
        }
        ++bench_run;
    }) << "ms\n";

    std::cout << "CUDA GPU Events average time: " << gpu_time_sum / (BENCHMARK_RUNS-1) << "ms\n";

    CU_ERR_CHECK_MSG(cudaEventDestroy(start_event), "");
    CU_ERR_CHECK_MSG(cudaEventDestroy(end_event), "");

    // Retrieve device data back to host
    cudaMemcpy2DFromArray(data,
        FFT_SIZE * 2*sizeof(float),
        (LOG_SIZE % 2 == 0) ? gpu_pp0.array : gpu_pp1.array,
        0, 0, FFT_SIZE * 2*sizeof(float), FFT_SIZE,
        cudaMemcpyDeviceToHost);

    // Print results
    // CUDA
    // std::cout << "========= CUDA =========\n";
    // for(size_t i = 0; i < FFT_SIZE*FFT_SIZE; ++i) {
    //     std::cout << "(" << data[i*2] << ", " << data[i*2+1] << ")" << "\n";
    // }

    // cuFFT
    // std::cout << "========= cuFFT =========\n";
    // for(size_t i = 0; i < FFT_SIZE*FFT_SIZE; ++i) {
    //     std::cout << "(" << cufft_data[i].x << ", " << cufft_data[i].y << ")" << "\n";
    // }

    // Free allocated resources
    destroy_surface(gpu_pp0);
    destroy_surface(gpu_pp1);
    free(data);
    return 0;
}