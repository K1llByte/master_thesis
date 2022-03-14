#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <cufft.h>
#include <cuda.h>

#define NX 256
#define BATCH 1

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
    printf("Hello World!\n");
    
    size_t data_size = sizeof(cufftComplex)*NX*BATCH;
    cufftComplex* data = (cufftComplex*) malloc(data_size);
    cufftComplex* gpu_data;
    cudaError_t err;
    // Allocate GPU buffer
    if((err = cudaMalloc((void**) &gpu_data, data_size)) != cudaSuccess)
    {
        fprintf(stderr, "Cuda error: Failed to allocate '%d'\n", err);
        return 1;
    }

    // Initializing input sequence
    for(size_t i = 0; i < NX; ++i)
    {
        data[i].x = i;
        data[i].y = 0.;
    }

    print_sequence(data);

    printf("%p %p %zu %d\n", gpu_data, data, sizeof(cufftComplex), cudaMemcpyHostToDevice);

    // Allocate GPU buffer
    if((err = cudaMemcpy(gpu_data, data, data_size, cudaMemcpyHostToDevice)) != cudaSuccess)
    {
        fprintf(stderr, "Cuda error: Failed to copy buffer to GPU '%d'\n", err);
        return 1;
    }

    cufftHandle plan;
    cufftResult_t res;
    if((res = cufftPlan1d(&plan, NX, CUFFT_C2C, BATCH)) != CUFFT_SUCCESS)
    {
        fprintf(stderr, "CUFFT error: Plan creation failed '%d'\n", res);
        return 1;
    }
    
    /* Note:
    * Identical pointers to input and output arrays implies in-place
    transformation
    */
    if((res = cufftExecC2C(plan, gpu_data, gpu_data, CUFFT_FORWARD)) != CUFFT_SUCCESS)
    {
        fprintf(stderr, "CUFFT error: ExecC2C Forward failed '%d'\n", err);
        return 1;
    }

    if ((err = cudaDeviceSynchronize()) != cudaSuccess)
    {
        fprintf(stderr, "Cuda error: Failed to synchronize\n");
        return;	
    }

    // // Print Computed FFT
    // printf("FFT:\n");
    // for(size_t i = 0; i < NX; ++i)
    // {
    //     printf("%f\n", data[i].x);
    // }
    
    if((res = cufftExecC2C(plan, gpu_data, gpu_data, CUFFT_INVERSE)) != CUFFT_SUCCESS)
    {
        fprintf(stderr, "CUFFT error: ExecC2C Inverse failed '%d'\n", err);
        return 1;
    }

    // Retrieve computed FFT buffer
    if((err = cudaMemcpy(data, gpu_data, data_size, cudaMemcpyHostToDevice)) != cudaSuccess)
    {
        fprintf(stderr, "Cuda error: Failed to copy buffer to GPU '%d'\n", err);
        return 1;
    }

    // // Print Computed IFFT
    // printf("FFT:\n");
    // for(size_t i = 0; i < NX; ++i)
    // {
    //     printf("%f\n", data[i].x);
    // }

    cufftDestroy(plan);
    cudaFree(data);
}