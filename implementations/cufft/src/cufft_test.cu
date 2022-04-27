#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <iostream>

#include <cufft.h>
#include <cuda.h>

#define NX 8
#define BATCH 1

class Foo;

// #define CU_ERR_CHECK_MSG(err, msg) {               \
//             if(err != cudaSuccess) {               \
//                 fprintf(stderr, msg);              \
//                 exit(1);                           \
//             }                                      \
//         }

// #define CU_CHECK_MSG(res, msg) {                   \
//             if(res != CUFFT_SUCCESS) {             \
//                 fprintf(stderr, msg);              \
//                 exit(1);                           \
//             }                                      \
//         }

#define CU_ERR_CHECK_MSG(err, msg, ...) {          \
            if(err != cudaSuccess) {               \
                fprintf(stderr, msg __VA_OPT__(,)  \
                    __VA_ARGS__);                  \
                exit(1);                           \
            }                                      \
        }

#define CU_CHECK_MSG(res, msg, ...) {              \
            if(res != CUFFT_SUCCESS) {             \
                fprintf(stderr, msg __VA_OPT__(,)  \
                    __VA_ARGS__);                  \
                exit(1);                           \
            }                                      \
        }


void print_sequence(const cufftComplex* arr)
{
    printf("[");
    for(size_t i = 0; i < NX; ++i)
    {
        printf((i == NX-1)
            ? "(%f + %fi)]\n"
            : "(%f + %fi), ", arr[i].x, arr[i].y);
    }
}

int main()
{
    
    size_t data_size = sizeof(cufftComplex)*NX*BATCH;
    cufftComplex* data = (cufftComplex*) malloc(data_size);
    cufftComplex* gpu_data;
    cudaError_t err;
    
    // Allocate GPU buffer
    err = cudaMalloc(&gpu_data, data_size);
    CU_ERR_CHECK_MSG(err, "Cuda error: Failed to allocate '%d'\n", err);

    // Initializing input sequence
    for(size_t i = 0; i < NX; ++i)
    {
        data[i].x = i;
        data[i].y = 0.;
    }

    print_sequence(data);

    // Copy data to GPU buffer
    err = cudaMemcpy(gpu_data, data, data_size, cudaMemcpyHostToDevice);
    CU_ERR_CHECK_MSG(err, "Cuda error: Failed to copy buffer to GPU '%d'\n", err);
    // CU_ERR_CHECK_MSG(err, "Cuda error: Failed to copy buffer to GPU\n");

    cufftHandle plan;
    cufftResult_t res;
    res = cufftPlan1d(&plan, NX, CUFFT_C2C, BATCH);
    CU_CHECK_MSG(res, "CUFFT error: Plan creation failed '%d'\n", res);
    // CU_CHECK_MSG(res, "CUFFT error: Plan creation failed\n");
    
    /* Note:
    * Identical pointers to input and output arrays implies in-place
    transformation
    */

    // Execute Forward 1D FFT
    res = cufftExecC2C(plan, gpu_data, gpu_data, CUFFT_FORWARD);
    CU_CHECK_MSG(res, "CUFFT error: ExecC2C Forward failed '%d'\n", res);
    // CU_CHECK_MSG(res, "CUFFT error: ExecC2C Forward failed\n");

    // Await end of execution
    err = cudaDeviceSynchronize();
    CU_ERR_CHECK_MSG(err, "Cuda error: Failed to synchronize\n");

    // Execute Inverse 1D FFT
    res = cufftExecC2C(plan, gpu_data, gpu_data, CUFFT_INVERSE);
    CU_CHECK_MSG(res, "CUFFT error: ExecC2C Inverse failed '%d'\n", res);
    // CU_CHECK_MSG(res, "CUFFT error: ExecC2C Inverse failed\n");

    // Await end of execution
    err = cudaDeviceSynchronize();
    CU_ERR_CHECK_MSG(err, "Cuda error: Failed to synchronize\n");

    // Retrieve computed FFT buffer
    err = cudaMemcpy(data, gpu_data, data_size, cudaMemcpyDeviceToHost);
    CU_ERR_CHECK_MSG(err, "Cuda error: Failed to copy buffer to GPU '%d'\n", err);

    // Divide result by N
    for(size_t i = 0; i < NX; ++i) {
        data[i].x /= NX;
    }


    cufftDestroy(plan);
    cudaFree(gpu_data);

    // Print Computed IFFT
    print_sequence(data);
}