#include <iostream>

#include "cuda_helper.hpp"

#define FFT_SIZE 256
#define BATCH 1
#define BENCHMARK_RUNS 2

// #define CU_ERR_CHECK_MSG(err, msg, ...) {  \
//     if(err != cudaSuccess) {               \
//         fprintf(stderr, msg __VA_OPT__(,)  \
//             __VA_ARGS__);                  \
//         exit(1);                           \
//     }                                      \
// }

// Actual kernel functions

#define CU_ERR_CHECK_MSG(err, msg) {  \
    if(err != cudaSuccess) {          \
        fprintf(stderr, msg);         \
        exit(1);                      \
    }                                 \
}

__device__ __forceinline__
float2 complex_mult(float2 v0, float2 v1) {
    return float2{
        v0.x * v1.x - v0.y * v1.y,
        v0.x * v1.y + v0.y * v1.x
    };
}

__device__ __forceinline__
float2 euler(float angle) {
    return float2{cos(angle), sin(angle)};
}


// GPU Kernel
__global__
void stockham_fft(int n, float a, float2* pingpong0, float2* pingpong1) {
    int line = blockIdx.x * blockDim.x + threadIdx.x;
    int column = blockIdx.y
    int pingpong = 0;

    // Continue ...
    
    if(i < n) {
        pingpong0[i] = a*pingpong0[i] + pingpong1[i];
    }
}

// // Host code
// int main() {
//     int N = 1 << 20;

//     // Allocate host memory
//     float *x, *y, *d_x, *d_y;
//     x = reinterpret_cast<float*>(malloc(FFT_SIZE*FFT_SIZE*sizeof(float)));
//     y = reinterpret_cast<float*>(malloc(FFT_SIZE*sizeof(float)));

//     // Allocate device memory
//     cudaError_t err = cudaMalloc(&d_x, N*sizeof(float)); 
//     CU_ERR_CHECK_MSG(err, "Cuda error: Failed to allocate\n");
//     err = cudaMalloc(&d_y, N*sizeof(float));
//     CU_ERR_CHECK_MSG(err, "Cuda error: Failed to allocate\n");
  
//     for (int i = 0; i < N; i++) {
//         x[i] = 1.0f;
//         y[i] = 2.0f;    
//     }

//     // Upload host data to device
//     err = cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
//     CU_ERR_CHECK_MSG(err, "Cuda error: Failed to copy buffer to GPU\n");
//     err = cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);
//     CU_ERR_CHECK_MSG(err, "Cuda error: Failed to copy buffer to GPU\n");
    
//     // Compute FFT on 1M elements
//     stockham_fft<<<(N+255)/256, 256>>>(N, 2.0f, d_x, d_y);

//     float maxError = 0.0f;
//     for (int i = 0; i < N; i++) {
//         maxError = max(maxError, abs(y[i]-4.0f));
//     }

//     std::cout << "Max error: " << maxError << "\n";

//     err = cudaDeviceSynchronize();
//     CU_ERR_CHECK_MSG(err, "Cuda error: Failed to synchronize\n");
    
//     // Retrieve device data back to host
//     err = cudaMemcpy(x, d_x, N*sizeof(float), cudaMemcpyDeviceToHost);
//     CU_ERR_CHECK_MSG(err, "Cuda error: Failed to copy buffer from GPU\n");
//     err = cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);
//     CU_ERR_CHECK_MSG(err, "Cuda error: Failed to copy buffer from GPU\n");


//     maxError = 0.0f;
//     for (int i = 0; i < N; i++) {
//         maxError = max(maxError, abs(y[i]-4.0f));
//     }
//     std::cout << "Max error: " << maxError << "\n";

//     cudaFree(d_x);
//     cudaFree(d_y);
//     free(x);
//     free(y);
// }


// Host code
int main() {
    const size_t BUFFER_SIZE_BYTES = FFT_SIZE*FFT_SIZE*sizeof(float2);

    // Allocate host memory
    float2* data;
    float2* gpu_pingpong0;
    float2* gpu_pingpong1;
    cudaError_t err;
    data = reinterpret_cast<float2*>(malloc(BUFFER_SIZE_BYTES));

    // Allocate device memory
    err = cudaMalloc(&gpu_pingpong0, BUFFER_SIZE_BYTES);
    CU_ERR_CHECK_MSG(err, "Cuda error: Failed to allocate\n");
    err = cudaMalloc(&gpu_pingpong1, BUFFER_SIZE_BYTES);
    CU_ERR_CHECK_MSG(err, "Cuda error: Failed to allocate\n");
    
    for (size_t i = 0; i < FFT_SIZE*FFT_SIZE; i++) {
        data[i].x = static_cast<float>(i);
    }
    
    // Upload host data to device
    err = cudaMemcpy(gpu_pingpong0, data, BUFFER_SIZE_BYTES, cudaMemcpyHostToDevice);
    CU_ERR_CHECK_MSG(err, "Cuda error: Failed to copy buffer to GPU\n");
    
    std::cout << "Success\n";
    // Compute FFT on 1M elements
    // stockham_fft<<<(N+255)/256, 256>>>(N, 2.0f, d_x, d_y);

    // float maxError = 0.0f;
    // for (int i = 0; i < N; i++) {
    //     maxError = max(maxError, abs(y[i]-4.0f));
    // }

    // std::cout << "Max error: " << maxError << "\n";

    // err = cudaDeviceSynchronize();
    // CU_ERR_CHECK_MSG(err, "Cuda error: Failed to synchronize\n");
    
    // // Retrieve device data back to host
    // err = cudaMemcpy(x, d_x, N*sizeof(float), cudaMemcpyDeviceToHost);
    // CU_ERR_CHECK_MSG(err, "Cuda error: Failed to copy buffer from GPU\n");
    // err = cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);
    // CU_ERR_CHECK_MSG(err, "Cuda error: Failed to copy buffer from GPU\n");


    // maxError = 0.0f;
    // for (int i = 0; i < N; i++) {
    //     maxError = max(maxError, abs(y[i]-4.0f));
    // }
    // std::cout << "Max error: " << maxError << "\n";

    // cudaFree(d_x);
    // cudaFree(d_y);
    // free(x);
    // free(y);
}