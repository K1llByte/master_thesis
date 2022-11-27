#include <iostream>

#include "dft.hpp"
#include "fft.hpp"
#include "fft_dit_iter.hpp"
#include "fft_dif_iter.hpp"
// #include "fft_dit_parallel.hpp"

// #include "fft_dif_nat_order.hpp"
// #include "fft_stockham.hpp"

// #include <ctime>
#include <chrono>
#include <functional>

// Auxiliar function wrapper to benchmark time execution
inline double benchmark(std::function<void()> func)
{
    auto begin = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    auto cpu_time = std::chrono::duration<double, std::milli>(end-begin).count();
    // std::cout << "Time: " << cpu_time << " ms\n";
    return cpu_time;
}

double average_benchmark(std::function<void()> func, const std::size_t average_count)
{
    auto time = 0.;
    for(std::size_t i = 0; i < average_count; ++i)
    {
        time += benchmark(func);
    }
    return time / average_count;
}


template<std::size_t N>
constexpr std::array<float, N> sample()
{
    std::array<float, N> res;
    for(std::size_t i = 0; i < N; ++i)
    {
        res[i] = i;
    }
    return res;
}

template<std::size_t N>
constexpr std::array<vec2, N> sample_complex()
{
    std::array<vec2, N> res;
    for(std::size_t i = 0; i < N; ++i)
    {
        res[i] = vec2{static_cast<float>(i), 0};
    }
    return res;
}


// template<std::size_t N>
// void sample_benchmark()
// {
//     std::array<double, 4> results;
//     auto test = sample<N>();
//     auto test_complex = sample_complex<N>();
//     const std::size_t average_count = 10;

//     // ====== DFT ====== //
//     results[0] = average_benchmark([&]{
//         auto freq = dft::fft(test);
//         auto val = dft::ifft(freq);
//     }, average_count);

//     // ====== DIT ====== //
//     results[1] = average_benchmark([&]{
//         auto freq = dit::fft(test);
//         dit::ifft(freq);
//     }, average_count);


//     // ====== DIF ====== //
//     results[2] = average_benchmark([&]{
//         auto freq = dif::fft(test_complex);
//         dif::ifft(freq);
//     }, average_count);

//     // ====== Recursive FFT ====== //
//     results[3] = average_benchmark([&]{
//         auto freq = recursive::fft(test);
//         recursive::ifft(freq);
//     }, average_count);

//     std::cout << "DFT: " << results[0] << '\n';
//     std::cout << "DIT: " << results[1] << '\n';
//     std::cout << "DIF: " << results[2] << '\n';
//     std::cout << "Rec FFT: " << results[3] << '\n';
// }

// void benchmark_all()
// {
//     std::cout << "=== Size 128 sample ===\n";
//     sample_benchmark<128>();
//     std::cout << "=== Size 256 sample ===\n";
//     sample_benchmark<256>();
//     std::cout << "=== Size 512 sample ===\n";
//     sample_benchmark<512>();
//     std::cout << "=== Size 1024 sample ===\n";
//     sample_benchmark<1024>();
//     std::cout << "=== Size 2048 sample ===\n";
//     sample_benchmark<2048>();
// }

// int main()
// {
//     auto time = 0.;
//     // const auto CYCLES = 128;
//     // for(auto i = 0; i < CYCLES; ++i)
//     // {
        
//         // using namespace recursive;
//         auto input = sample<4>();
//         auto complex_input = sample_complex<4>();
//         // std::array<float, 4>{0, 1, 2, 3};

//         // time += benchmark([&]{
//         //     auto freq = recursive::fft(input);
//         //     // print_complex_array(freq);
//         //     auto val = recursive::ifft(freq);
//         //     // print_complex_array(val);
//         // });

//         benchmark([&]{
//             auto freq = dft::fft(input);
//             // print_complex_array(freq);
//             auto val = dft::ifft(freq);
//             print_complex_array(val);
//         });

//         // benchmark([&]{
//         //     auto freq = dif::fft(complex_input);
//         //     // print_complex_array(freq);
//         //     auto val = dif::ifft(freq);
//         //     print_complex_array(val);
//         // });

//     // }
// }

// int main()
// {
//     benchmark_all();
// }

int main()
{
    constexpr auto N = 4;
    auto input = sample<N>();
    ////////////// Benchmark //////////////

    // auto input_complex = to_complex(input);
    // auto time1 = benchmark([&]{
    //     dit::parallel::fft(input_complex);
    // });

    // auto time2 = benchmark([&]{
    //     dit::fft(input);
    // });


    // std::cout << "Parallel time: " << time1 << "\n";
    // std::cout << "Sequential time: " << time2 << "\n";

    ////////////// Test //////////////

    // auto freq1 = dit::parallel::fft(to_complex(input));
    // std::cout << "FFT: ";
    // print_complex_array(freq1);
    // auto val1 = dit::parallel::ifft(freq1);
    // std::cout << "IFFT: ";
    // print_complex_array(val1);

    // Used to compare results
    
    std::array<float,4> in{0,4,8,12};
    auto real_freq1 = dit::fft(in);
    print_array(in);
    print_complex_array(real_freq1);
    // auto val1 = stockham::ifft(real_freq1);
    // std::cout << "EXPECTED: ";
    // print_array(input);
    // print_complex_array(real_freq1);
    // print_complex_array(val1);
    // std::cout << "ARE EQUAL? " << equal_complex_arrays(real_freq1, freq1) << "\n";
    // std::cout << "ARE EQUAL? " << equal_complex_arrays(to_complex(input), val1) << "\n";

    // auto output2 = dit::fft(input);
    // // std::cout << "EXPECTED: ";
    // // print_complex_array(output2);
    // std::cout << "ARE EQUAL? " << equal_complex_arrays(output1, output2) << "\n";

    // auto output1 = input;
    // auto output2 = input;
    // inplace_bit_reversal(output1);

    // for(std::size_t i = 0 ; i < N ; ++i)
    // {
    //     output2[bit_reverse(i, static_cast<std::size_t>(std::log2(N)))] = input[i];
    // }
    // std::cout << "COMPUTED: ";
    // print_array(output1);
    // std::cout << "EXPECTED: ";
    // print_array(output2);
    // std::cout << "ARE EQUAL? " << equal_float_arrays(output1, output2) << "\n";
}
