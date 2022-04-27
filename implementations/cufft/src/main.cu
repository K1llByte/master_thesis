#include <iostream>
#include <array>

#include <chrono>
#include <functional>

#include <cufft.h>
#include <cuda.h>

#define FFT_SIZE 8
#define BATCH 1

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


// Auxiliar function wrapper to benchmark time execution
inline double benchmark(std::function<void()> func)
{
    auto begin = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    auto cpu_time = std::chrono::duration<double, std::milli>(end-begin).count();
    std::cout << "Time: " << cpu_time << " ms\n";
    return cpu_time;
}

void print_sequence(const cufftComplex* arr)
{
    printf("[");
    for(size_t i = 0; i < FFT_SIZE; ++i)
    {
        printf((i == FFT_SIZE-1)
            ? "(%.2f + %.2fi)]\n"
            : "(%.2f + %.2fi), ", arr[i].x, arr[i].y);
    }
}

void print_matrix(const cufftComplex* arr)
{
    for(size_t i = 0; i < FFT_SIZE; ++i) {
        printf("[");
        for(size_t j = 0; j < FFT_SIZE; ++j) {
            printf((j == FFT_SIZE-1)
                ? "(%.2f + %.2fi)]\n"
                : "(%.2f + %.2fi), ", arr[i*FFT_SIZE+j].x, arr[i*FFT_SIZE+j].y);
        }
    }
}

void compute_fft()
{
    const size_t data_size = sizeof(cufftComplex)*FFT_SIZE*FFT_SIZE;
    cufftComplex* data = reinterpret_cast<cufftComplex*>(malloc(data_size));
    cufftComplex* gpu_data;
    cudaError_t err;

    // Initializing input sequence
    for(size_t i = 0; i < FFT_SIZE; ++i) {
        for(size_t j = 0; j < FFT_SIZE; ++j) {
            data[i*FFT_SIZE + j].x = i*FFT_SIZE + j;
            data[i*FFT_SIZE + j].y = 0.;
        }
    }

    // Print input
    printf("INPUT: ");
    print_matrix(data);

    // Allocate GPU buffer
    err = cudaMalloc(&gpu_data, data_size);
    CU_ERR_CHECK_MSG(err, "Cuda error: Failed to allocate '%d'\n", err);

    // Copy data to GPU buffer
    err = cudaMemcpy(gpu_data, data, data_size, cudaMemcpyHostToDevice);
    CU_ERR_CHECK_MSG(err, "Cuda error: Failed to copy buffer to GPU '%d'\n", err);

    // Setup cufft plan
    cufftHandle plan;
    cufftResult_t res;
    res = cufftPlan2d(&plan, FFT_SIZE, FFT_SIZE, CUFFT_C2C);
    CU_CHECK_MSG(res, "cuFFT error: Plan creation failed '%d'\n", res);

    // Execute Forward 1D FFT
    res = cufftExecC2C(plan, gpu_data, gpu_data, CUFFT_FORWARD);
    CU_CHECK_MSG(res, "cuFFT error: ExecC2C Forward failed '%d'\n", res);

    // Await end of execution
    err = cudaDeviceSynchronize();
    CU_ERR_CHECK_MSG(err, "Cuda error: Failed to synchronize\n");

    // Retrieve computed FFT buffer
    err = cudaMemcpy(data, gpu_data, data_size, cudaMemcpyDeviceToHost);
    CU_ERR_CHECK_MSG(err, "Cuda error: Failed to copy buffer to GPU '%d'\n", err);

    // Divide result by N
    for(size_t i = 0; i < FFT_SIZE; ++i) {
        data[i].x /= FFT_SIZE;
    }

    // Print computed output
    printf("OUTPUT: ");
    print_matrix(data);

    // Destroy Cuda and cuFFT context
    cufftDestroy(plan);
    cudaFree(gpu_data);
}

int main()
{
    benchmark(compute_fft);
}