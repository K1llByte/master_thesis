#include <iostream>
#include <array>

#include <chrono>
#include <functional>

#include <cufft.h>
#include <cuda.h>

#define FFT_SIZE 256
// #define BATCH
#define BENCHMARK_RUNS 30

// #define CU_ERR_CHECK_MSG(err, msg, ...) {          \
//             if(err != cudaSuccess) {               \
//                 fprintf(stderr, msg __VA_OPT__(,)  \
//                     __VA_ARGS__);                  \
//                 exit(1);                           \
//             }                                      \
//         }

// #define CU_CHECK_MSG(res, msg, ...) {              \
//             if(res != CUFFT_SUCCESS) {             \
//                 fprintf(stderr, msg __VA_OPT__(,)  \
//                     __VA_ARGS__);                  \
//                 exit(1);                           \
//             }                                      \
//         }

#define CU_ERR_CHECK_MSG(err, msg) {               \
            if(err != cudaSuccess) {               \
                fprintf(stderr, msg);              \
                exit(1);                           \
            }                                      \
        }

#define CU_CHECK_MSG(res, msg) {                   \
            if(res != CUFFT_SUCCESS) {             \
                fprintf(stderr, msg);              \
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
    //std::cout << "Time: " << cpu_time << " ms\n";
    return cpu_time;
}

void print_sequence(cufftComplex const *const  arr)
{
    printf("[");
    for(size_t i = 0; i < FFT_SIZE; ++i)
    {
        printf((i == FFT_SIZE-1)
            ? "(%.2f + %.2fi)]\n"
            : "(%.2f + %.2fi), ", arr[i].x, arr[i].y);
    }
}

void print_matrix(cufftComplex const *const  arr)
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

////////////////////////////////////////////////////

void compute_2d_fft_batch()
{
    const int batch_size = 2;
    const size_t data_size = sizeof(cufftComplex)*FFT_SIZE*FFT_SIZE*batch_size;
    cufftComplex* data = reinterpret_cast<cufftComplex*>(malloc(data_size));
    cufftComplex* gpu_data_in;
    cufftComplex* gpu_data_out;
    cudaError_t err;

    // Initializing input sequence
    for(size_t i = 0; i < FFT_SIZE*FFT_SIZE*batch_size; ++i) {
        data[i].x = i;
        data[i].y = i;
    }

    // Print input
    // printf("INPUT: ");
    // print_matrix(data);

    // Allocate Input GPU buffer
    err = cudaMalloc(&gpu_data_in, data_size);
    CU_ERR_CHECK_MSG(err, "Cuda error: Failed to allocate\n");

    // Allocate Output GPU buffer
    err = cudaMalloc(&gpu_data_out, data_size);
    CU_ERR_CHECK_MSG(err, "Cuda error: Failed to allocate\n");

    // Copy data to GPU input buffer
    err = cudaMemcpy(gpu_data_in, data, data_size, cudaMemcpyHostToDevice);
    CU_ERR_CHECK_MSG(err, "Cuda error: Failed to copy buffer to GPU\n");

    // Setup cufft plan
    cufftHandle plan;
    cufftResult_t res;
    // res = cufftPlan2d(&plan, FFT_SIZE, FFT_SIZE, CUFFT_C2C);
    int fft_size[] = {FFT_SIZE, FFT_SIZE};
    res = cufftPlanMany(&plan, 2, fft_size,
        nullptr, 0, 0,
        nullptr, 0, 0,
        CUFFT_C2C, batch_size
    );
    CU_CHECK_MSG(res, "cuFFT error: Plan creation failed\n");

    // Benchmark Forward FFT
    double forward_time = 0.;
    for(int i = 0; i < BENCHMARK_RUNS ; ++i) {
        forward_time += benchmark([&]{
            // Execute Forward 1D FFT
            res = cufftExecC2C(plan, gpu_data_in, gpu_data_out, CUFFT_FORWARD);
            CU_CHECK_MSG(res, "cuFFT error: ExecC2C Forward failed\n");
    
            // Await end of execution
            err = cudaDeviceSynchronize();
            CU_ERR_CHECK_MSG(err, "Cuda error: Failed to synchronize\n");
        });
    }
    std::cout << "Forward: " << forward_time / BENCHMARK_RUNS << "ms\n";
    // std::cout << "Forward: " << forward_time << "ms\n";

    // Benchmark Inverse FFT
    double inverse_time = 0.;
    for(int i = 0; i < BENCHMARK_RUNS ; ++i) {
        inverse_time += benchmark([&]{
            // Execute Forward 1D FFT
            res = cufftExecC2C(plan, gpu_data_in, gpu_data_out, CUFFT_FORWARD);
            CU_CHECK_MSG(res, "cuFFT error: ExecC2C Forward failed\n");
    
            // Await end of execution
            err = cudaDeviceSynchronize();
            CU_ERR_CHECK_MSG(err, "Cuda error: Failed to synchronize\n");
        });
    }
    std::cout << "Inverse: " << inverse_time / BENCHMARK_RUNS << "ms\n";

    // Retrieve computed FFT buffer
    err = cudaMemcpy(data, gpu_data_in, data_size, cudaMemcpyDeviceToHost);
    CU_ERR_CHECK_MSG(err, "Cuda error: Failed to copy buffer to GPU\n");

    // Divide result by N
    // for(size_t i = 0; i < FFT_SIZE*FFT_SIZE; ++i) {
    //     data[i].x /= (FFT_SIZE*FFT_SIZE);
    //     data[i].y /= (FFT_SIZE*FFT_SIZE);
    // }

    // Print computed Inverse FFT output
    printf("Inverse FFT: \n");
    // print_matrix(data);
    // print_matrix(data + FFT_SIZE*FFT_SIZE);

    // Destroy Cuda and cuFFT context
    cufftDestroy(plan);
    cudaFree(gpu_data_in);
}



void compute_2d_fft()
{
    const size_t data_size = sizeof(cufftComplex)*FFT_SIZE*FFT_SIZE;
    cufftComplex* data = reinterpret_cast<cufftComplex*>(malloc(data_size));
    cufftComplex* gpu_data_in;
    cufftComplex* gpu_data_out;
    cudaError_t err;

    // Initializing input sequence
    for(size_t i = 0; i < FFT_SIZE*FFT_SIZE; ++i) {
        data[i].x = i;
        data[i].y = FFT_SIZE*FFT_SIZE - i - 1;
    }

    // Print input
    // printf("INPUT: ");
    // print_matrix(data);

    // Allocate Input GPU buffer
    err = cudaMalloc(&gpu_data_in, data_size);
    // CU_ERR_CHECK_MSG(err, "Cuda error: Failed to allocate '%d'\n", err);
    CU_ERR_CHECK_MSG(err, "Cuda error: Failed to allocate\n");

    // Allocate Output GPU buffer
    err = cudaMalloc(&gpu_data_out, data_size);
    // CU_ERR_CHECK_MSG(err, "Cuda error: Failed to allocate '%d'\n", err);
    CU_ERR_CHECK_MSG(err, "Cuda error: Failed to allocate\n");

    // Copy data to GPU buffer
    err = cudaMemcpy(gpu_data_in, data, data_size, cudaMemcpyHostToDevice);
    // CU_ERR_CHECK_MSG(err, "Cuda error: Failed to copy buffer to GPU '%d'\n", err);
    CU_ERR_CHECK_MSG(err, "Cuda error: Failed to copy buffer to GPU\n");

    // Setup cufft plan
    cufftHandle plan;
    cufftResult_t res;
    res = cufftPlan2d(&plan, FFT_SIZE, FFT_SIZE, CUFFT_C2C);
    // CU_CHECK_MSG(res, "cuFFT error: Plan creation failed '%d'\n", res);
    CU_CHECK_MSG(res, "cuFFT error: Plan creation failed\n");

    // Benchmark Forward FFT
    double forward_time = 0.;
    for(int i = 0; i < BENCHMARK_RUNS ; ++i) {
        forward_time += benchmark([&]{
            // Execute Forward 1D FFT
            res = cufftExecC2C(plan, gpu_data_in, gpu_data_out, CUFFT_FORWARD);
            // CU_CHECK_MSG(res, "cuFFT error: ExecC2C Forward failed '%d'\n", res);
            CU_CHECK_MSG(res, "cuFFT error: ExecC2C Forward failed\n");
    
            // Await end of execution
            err = cudaDeviceSynchronize();
            CU_ERR_CHECK_MSG(err, "Cuda error: Failed to synchronize\n");
        });
    }
    std::cout << "Forward: " << forward_time / BENCHMARK_RUNS << "ms\n";
    // std::cout << "Forward: " << forward_time << "ms\n";

    // Benchmark Inverse FFT
    double inverse_time = 0.;
    for(int i = 0; i < BENCHMARK_RUNS ; ++i) {
        inverse_time += benchmark([&]{
            // Execute Forward 1D FFT
            res = cufftExecC2C(plan, gpu_data_in, gpu_data_out, CUFFT_FORWARD);
            // CU_CHECK_MSG(res, "cuFFT error: ExecC2C Forward failed '%d'\n", res);
            CU_CHECK_MSG(res, "cuFFT error: ExecC2C Forward failed\n");
    
            // Await end of execution
            err = cudaDeviceSynchronize();
            CU_ERR_CHECK_MSG(err, "Cuda error: Failed to synchronize\n");
        });
    }
    std::cout << "Inverse: " << inverse_time / BENCHMARK_RUNS << "ms\n";

    // Retrieve computed FFT buffer
    err = cudaMemcpy(data, gpu_data_in, data_size, cudaMemcpyDeviceToHost);
    // CU_ERR_CHECK_MSG(err, "Cuda error: Failed to copy buffer to GPU '%d'\n", err);
    CU_ERR_CHECK_MSG(err, "Cuda error: Failed to copy buffer to GPU\n");

    // Divide result by N
    // for(size_t i = 0; i < FFT_SIZE*FFT_SIZE; ++i) {
    //     data[i].x /= (FFT_SIZE*FFT_SIZE);
    //     data[i].y /= (FFT_SIZE*FFT_SIZE);
    // }

    // Print computed Inverse FFT output
    printf("Inverse FFT: \n");
    // print_matrix(data);

    // Destroy Cuda and cuFFT context
    cufftDestroy(plan);
    cudaFree(gpu_data_in);
}

void compute_2d_fft_in_place()
{
    const size_t data_size = sizeof(cufftComplex)*FFT_SIZE*FFT_SIZE;
    cufftComplex* data = reinterpret_cast<cufftComplex*>(malloc(data_size));
    cufftComplex* gpu_data;
    cudaError_t err;

    // Initializing input sequence
    for(size_t i = 0; i < FFT_SIZE*FFT_SIZE; ++i) {
        data[i].x = i;
        data[i].y = 0.00;
    }

    // Print input
    // printf("INPUT: ");
    // print_matrix(data);

    // Allocate GPU buffer
    err = cudaMalloc(&gpu_data, data_size);
    // CU_ERR_CHECK_MSG(err, "Cuda error: Failed to allocate '%d'\n", err);
    CU_ERR_CHECK_MSG(err, "Cuda error: Failed to allocate\n");

    // Copy data to GPU buffer
    err = cudaMemcpy(gpu_data, data, data_size, cudaMemcpyHostToDevice);
    // CU_ERR_CHECK_MSG(err, "Cuda error: Failed to copy buffer to GPU '%d'\n", err);
    CU_ERR_CHECK_MSG(err, "Cuda error: Failed to copy buffer to GPU\n");

    // Setup cufft plan
    cufftHandle plan;
    cufftResult_t res;
    res = cufftPlan2d(&plan, FFT_SIZE, FFT_SIZE, CUFFT_C2C);
    // CU_CHECK_MSG(res, "cuFFT error: Plan creation failed '%d'\n", res);
    CU_CHECK_MSG(res, "cuFFT error: Plan creation failed\n");

    std::cout << "Forward\n";
    benchmark([&]{
        // Execute Forward 1D FFT
        res = cufftExecC2C(plan, gpu_data, gpu_data, CUFFT_FORWARD);
        // CU_CHECK_MSG(res, "cuFFT error: ExecC2C Forward failed '%d'\n", res);
        CU_CHECK_MSG(res, "cuFFT error: ExecC2C Forward failed\n");

        // Await end of execution
        err = cudaDeviceSynchronize();
        CU_ERR_CHECK_MSG(err, "Cuda error: Failed to synchronize\n");
    });

    std::cout << "Inverse\n";
    benchmark([&]{
        // Execute Inverse 1D FFT
        res = cufftExecC2C(plan, gpu_data, gpu_data, CUFFT_INVERSE);
        // CU_CHECK_MSG(res, "CUFFT error: ExecC2C Inverse failed '%d'\n", res);
        CU_CHECK_MSG(res, "CUFFT error: ExecC2C Inverse failed\n");

        // TODO: Check if this is necessary
        // Await end of execution
        err = cudaDeviceSynchronize();
        CU_ERR_CHECK_MSG(err, "Cuda error: Failed to synchronize\n");

    });
    // Retrieve computed FFT buffer
    err = cudaMemcpy(data, gpu_data, data_size, cudaMemcpyDeviceToHost);
    // CU_ERR_CHECK_MSG(err, "Cuda error: Failed to copy buffer to GPU '%d'\n", err);
    CU_ERR_CHECK_MSG(err, "Cuda error: Failed to copy buffer to GPU\n");

    // Divide result by N
    for(size_t i = 0; i < FFT_SIZE*FFT_SIZE; ++i) {
        data[i].x /= (FFT_SIZE*FFT_SIZE);
        data[i].y = 0;
    }

    // Print computed Inverse FFT output
    //printf("Inverse FFT: \n");
    //print_matrix(data);

    // Destroy Cuda and cuFFT context
    cufftDestroy(plan);
    cudaFree(gpu_data);
}

void compute_1d_fft()
{
    const size_t data_size = sizeof(cufftComplex)*FFT_SIZE;
    cufftComplex* data = reinterpret_cast<cufftComplex*>(malloc(data_size));
    cufftComplex* gpu_data;
    cudaError_t err;

    // Initializing input sequence
    for(size_t i = 0; i < FFT_SIZE; ++i) {
        data[i].x = i;
        data[i].y = 0.00;
    }

    // Print input
    // printf("INPUT: ");
    // print_matrix(data);

    // Allocate GPU buffer
    err = cudaMalloc(&gpu_data, data_size);
    // CU_ERR_CHECK_MSG(err, "Cuda error: Failed to allocate '%d'\n", err);
    CU_ERR_CHECK_MSG(err, "Cuda error: Failed to allocate\n");

    // Copy data to GPU buffer
    err = cudaMemcpy(gpu_data, data, data_size, cudaMemcpyHostToDevice);
    // CU_ERR_CHECK_MSG(err, "Cuda error: Failed to copy buffer to GPU '%d'\n", err);
    CU_ERR_CHECK_MSG(err, "Cuda error: Failed to copy buffer to GPU\n");

    // Setup cufft plan
    cufftHandle plan;
    cufftResult_t res;
    res = cufftPlan1d(&plan, FFT_SIZE, CUFFT_C2C, 1);
    // CU_CHECK_MSG(res, "cuFFT error: Plan creation failed '%d'\n", res);
    CU_CHECK_MSG(res, "cuFFT error: Plan creation failed\n");

    std::cout << "Forward\n";
    benchmark([&]{
        // Execute Forward 1D FFT
        res = cufftExecC2C(plan, gpu_data, gpu_data, CUFFT_FORWARD);
        // CU_CHECK_MSG(res, "cuFFT error: ExecC2C Forward failed '%d'\n", res);
        CU_CHECK_MSG(res, "cuFFT error: ExecC2C Forward failed\n");

        // Await end of execution
        err = cudaDeviceSynchronize();
        CU_ERR_CHECK_MSG(err, "Cuda error: Failed to synchronize\n");
    });

    std::cout << "Inverse\n";
    benchmark([&]{
        // Execute Inverse 1D FFT
        res = cufftExecC2C(plan, gpu_data, gpu_data, CUFFT_INVERSE);
        // CU_CHECK_MSG(res, "CUFFT error: ExecC2C Inverse failed '%d'\n", res);
        CU_CHECK_MSG(res, "CUFFT error: ExecC2C Inverse failed\n");

        // TODO: Check if this is necessary
        // Await end of execution
        err = cudaDeviceSynchronize();
        CU_ERR_CHECK_MSG(err, "Cuda error: Failed to synchronize\n");
    });

    // Retrieve computed Inverse FFT buffer
    err = cudaMemcpy(data, gpu_data, data_size, cudaMemcpyDeviceToHost);
    // CU_ERR_CHECK_MSG(err, "Cuda error: Failed to copy buffer to GPU '%d'\n", err);
    CU_ERR_CHECK_MSG(err, "Cuda error: Failed to copy buffer to GPU\n");

    // Divide result by N
    for(size_t i = 0; i < FFT_SIZE; ++i) {
        data[i].x /= FFT_SIZE;
        data[i].y = 0;
    }

    // Print computed output
    //printf("Inverse FFT: ");
    //print_sequence(data);

    // Destroy Cuda and cuFFT context
    cufftDestroy(plan);
    cudaFree(gpu_data);
}

void compute_1d_fft_in_place()
{
    const size_t data_size = sizeof(cufftComplex)*FFT_SIZE;
    cufftComplex* data = reinterpret_cast<cufftComplex*>(malloc(data_size));
    cufftComplex* gpu_data_in;
    cufftComplex* gpu_data_out;
    cudaError_t err;

    // Initializing input sequence
    for(size_t i = 0; i < FFT_SIZE; ++i) {
        data[i].x = i;
        data[i].y = 0.00;
    }

    // Print input
    // printf("INPUT: ");
    // print_matrix(data);

    // Allocate Input GPU buffer
    err = cudaMalloc(&gpu_data_in, data_size);
    // CU_ERR_CHECK_MSG(err, "Cuda error: Failed to allocate '%d'\n", err);
    CU_ERR_CHECK_MSG(err, "Cuda error: Failed to allocate\n");

    // Allocate Output GPU buffer
    err = cudaMalloc(&gpu_data_out, data_size);
    // CU_ERR_CHECK_MSG(err, "Cuda error: Failed to allocate '%d'\n", err);
    CU_ERR_CHECK_MSG(err, "Cuda error: Failed to allocate\n");

    // Copy data to GPU buffer
    err = cudaMemcpy(gpu_data_in, data, data_size, cudaMemcpyHostToDevice);
    // CU_ERR_CHECK_MSG(err, "Cuda error: Failed to copy buffer to GPU '%d'\n", err);
    CU_ERR_CHECK_MSG(err, "Cuda error: Failed to copy buffer to GPU\n");

    // Setup cufft plan
    cufftHandle plan;
    cufftResult_t res;
    res = cufftPlan1d(&plan, FFT_SIZE, CUFFT_C2C, 1);
    // CU_CHECK_MSG(res, "cuFFT error: Plan creation failed '%d'\n", res);
    CU_CHECK_MSG(res, "cuFFT error: Plan creation failed\n");

    std::cout << "Forward\n";
    benchmark([&]{
        // Execute Forward 1D FFT
        res = cufftExecC2C(plan, gpu_data_in, gpu_data_out, CUFFT_FORWARD);
        // CU_CHECK_MSG(res, "cuFFT error: ExecC2C Forward failed '%d'\n", res);
        CU_CHECK_MSG(res, "cuFFT error: ExecC2C Forward failed\n");

        // Await end of execution
        err = cudaDeviceSynchronize();
        CU_ERR_CHECK_MSG(err, "Cuda error: Failed to synchronize\n");
    });

    std::cout << "Inverse\n";
    benchmark([&]{
        // Execute Inverse 1D FFT
        res = cufftExecC2C(plan, gpu_data_out, gpu_data_in, CUFFT_INVERSE);
        // CU_CHECK_MSG(res, "CUFFT error: ExecC2C Inverse failed '%d'\n", res);
        CU_CHECK_MSG(res, "CUFFT error: ExecC2C Inverse failed\n");

        // TODO: Check if this is necessary
        // Await end of execution
        err = cudaDeviceSynchronize();
        CU_ERR_CHECK_MSG(err, "Cuda error: Failed to synchronize\n");
    });

    // Retrieve computed Inverse FFT buffer
    err = cudaMemcpy(data, gpu_data_in, data_size, cudaMemcpyDeviceToHost);
    // CU_ERR_CHECK_MSG(err, "Cuda error: Failed to copy buffer to GPU '%d'\n", err);
    CU_ERR_CHECK_MSG(err, "Cuda error: Failed to copy buffer to GPU\n");

    // Divide result by N
    for(size_t i = 0; i < FFT_SIZE; ++i) {
        data[i].x /= FFT_SIZE;
        data[i].y = 0;
    }

    // Print computed output
    //printf("Inverse FFT: ");
    //print_sequence(data);

    // Destroy Cuda and cuFFT context
    cufftDestroy(plan);
    cudaFree(gpu_data_in);
}

int main()
{
    #ifdef BATCH
        compute_2d_fft_batch();
    #else
        compute_2d_fft();
    #endif
    // compute_2d_fft_in_place();
}