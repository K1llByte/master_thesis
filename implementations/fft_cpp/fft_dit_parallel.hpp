#ifndef FFT_DIT_PARALLEL_HPP
#define FFT_DIT_PARALLEL_HPP

// #include <type_traits>
// #include <concepts>
#include <cstdint>
#include <vector>
#include <thread>
#include <barrier>
#include <mutex>
#include <sstream>

#include "utils.hpp"

namespace dit::parallel
{
    //////////////// Forward FFT Entry Point ////////////////

    template<std::size_t N>
    const std::array<vec2, N> fft(const std::array<vec2, N>& p)
    {
        constexpr auto log_n = static_cast<std::size_t>(std::log2(N));

        // Since this isn't an inlined version of the algorithm
        // we take O(N) space complexity with the Bit-reverse copy step
        std::array<vec2, N> res;
        std::array<vec2, N> pingpong;

        for(std::size_t i = 0 ; i < N ; ++i)
        {
            res[bit_reverse(i, log_n)] = p[i];
        }
        // inplace_bit_reversal(res);

        constexpr auto num_processors = 4;
        constexpr auto vals_per_processors = N / num_processors;

        if(N % num_processors != 0)
            throw std::runtime_error("Invalid config of threads");
        std::barrier sync_point(num_processors);
        std::mutex debug_lock;

        std::vector<std::thread> processors(num_processors);
        for(std::size_t i = 0; i < num_processors; ++i)
        {
            // Concurrent processors
            processors[i] = std::thread([&](std::size_t id)
            {
                for(std::size_t stage = 0; stage < log_n; ++stage)
                {
                    std::array<vec2, N>* from = &res;
                    std::array<vec2, N>* to = &pingpong;
                    if(stage % 2 != 0)
                    {
                        from = &pingpong;
                        to = &res;
                    }

                    // bool torf = false;
                    const std::size_t group_size = 2 << stage;
                    for(std::size_t val = 0; val < vals_per_processors; ++val)
                    {
                        const auto idx = id*vals_per_processors + val;
                        const auto half_group_size = group_size >> 1;
                        const auto w = euler(-2 * (pi / group_size) * ((idx % group_size) % half_group_size));
                        // debug_lock.lock();
                        // std::cout << w[0] << ", " << w[1] << "\n";
                        // debug_lock.unlock();

                        if((idx % group_size) >= half_group_size)
                        {
                            // const auto w = euler(-2 * pi / group_size * ((idx % group_size) % half_group_size));
                            // if(!torf)
                            // {
                            //     debug_lock.lock();
                            //     std::stringstream ss;
                            //     // ss << "ID: " << id << "\n";
                            //     // ss << "group_size: " << group_size << "\n";
                            //     ss << "idxs: " << idx-half_group_size << " - " << idx << "\n";
                            //     // ss << "w: " << "(" << w[0] << ", " << w[1] << ")" << "\n";
                            //     ss << "==================\n";
                            //     std::cout << ss.str();
                            //     debug_lock.unlock();
                            // }
                            (*to)[idx] = complex_sub((*from)[idx-half_group_size], complex_mul(w, (*from)[idx]));
                            // (*to)[idx] = complex_sub((*from)[idx-half_group_size], (*from)[idx]);
                        }
                        else
                        {
                            // const auto w = euler(-2 * pi / group_size * ((idx % group_size) % half_group_size));
                            // if(!torf)
                            // {
                            //     debug_lock.lock();
                            //     std::stringstream ss;
                            //     // ss << "ID: " << id << "\n";
                            //     // ss << "group_size: " << group_size << "\n";
                            //     ss << "idxs: " << idx << " + " << idx+half_group_size << "\n";
                            //     // ss << "w: " << "(" << w[0] << ", " << w[1] << ")" << "\n";
                            //     ss << "==================\n";
                            //     std::cout << ss.str();
                            //     debug_lock.unlock();
                            // }
                            (*to)[idx] = complex_add((*from)[idx], complex_mul(w, (*from)[idx+half_group_size]));
                            // (*to)[idx] = complex_add((*from)[idx], (*from)[idx+half_group_size]);
                        }

                    }
                    // printf("%zu\n", id);
                    sync_point.arrive_and_wait();

                    // debug_lock.lock();
                    // std::cout << "END ID: " << id << " -> ";
                    // if(id == 0)
                    // {
                    //     std::cout << "PARALEL " << stage+1 << ": ";
                    //     print_complex_array(fp_error_resolve(*to));
                    // }
                    // debug_lock.unlock();
                    // sync_point.arrive_and_wait();

                }
            }, i);
        }

        auto begin = std::chrono::high_resolution_clock::now();
        for(auto& processor : processors)
            processor.join();
        auto end = std::chrono::high_resolution_clock::now();
        auto cpu_time = std::chrono::duration<double, std::milli>(end-begin).count();
        std::cout << "Remove parallel time: " << cpu_time << "\n";
        
        return (log_n & 1) ? pingpong : res;
    }

    //////////////// Inverse FFT Entry Point ////////////////

    template<std::size_t N>
    const std::array<vec2, N> ifft(const std::array<vec2, N>& p)
    {
        constexpr auto log_n = static_cast<std::size_t>(std::log2(N));

        std::array<vec2, N> res;
        std::array<vec2, N> pingpong;

        for(std::size_t i = 0 ; i < N ; ++i)
        {
            res[bit_reverse(i, log_n)] = p[i];
        }

        constexpr auto num_processors = 4;
        constexpr auto vals_per_processors = N / num_processors;
        if(N % num_processors != 0)
            throw std::runtime_error("Invalid config of threads");
        std::barrier sync_point(num_processors);
        std::mutex debug_lock;
        constexpr float mult_factor = 1.0 / N;

        std::vector<std::thread> processors(num_processors);
        for(std::size_t i = 0; i < num_processors; ++i)
        {
            // Concurrent processors
            processors[i] = std::thread([&](std::size_t id)
            {
                for(std::size_t stage = 0; stage < log_n; ++stage)
                {
                    std::array<vec2, N>* from = &res;
                    std::array<vec2, N>* to = &pingpong;
                    if(stage % 2 != 0)
                    {
                        from = &pingpong;
                        to = &res;
                    }

                    // bool torf = false;
                    const std::size_t group_size = 2 << stage;
                    for(std::size_t val = 0; val < vals_per_processors; ++val)
                    {
                        const auto idx = id*vals_per_processors + val;
                        const auto half_group_size = group_size >> 1;
                        const auto w = euler(2 * (pi / group_size) * ((idx % group_size) % half_group_size));

                        if((idx % group_size) >= half_group_size)
                        {
                            (*to)[idx] = complex_sub((*from)[idx-half_group_size], complex_mul(w, (*from)[idx]));
                        }
                        else
                        {
                            (*to)[idx] = complex_add((*from)[idx], complex_mul(w, (*from)[idx+half_group_size]));
                        }
                        if(stage == log_n-1)
                            (*to)[idx] = vec2{(*to)[idx][0] * mult_factor, (*to)[idx][1] * mult_factor};

                    }
                    sync_point.arrive_and_wait();
                }

                // Divide each element with N
                // auto& final_buffer = (log_n & 1) ? pingpong : res;
                // for(std::size_t j = 0; j < vals_per_processors; ++j)
                // {
                //     const auto idx = id*vals_per_processors + j;
                //     final_buffer[idx] = vec2{
                //         final_buffer[idx][0] / N,
                //         final_buffer[idx][1] / N
                //     };
                // }
            }, i);
        }

        // auto begin = std::chrono::high_resolution_clock::now();
        for(auto& processor : processors)
            processor.join();
        // auto end = std::chrono::high_resolution_clock::now();
        // auto cpu_time = std::chrono::duration<double, std::milli>(end-begin).count();
        // std::cout << "Remove parallel time: " << cpu_time << "\n";
        
        return (log_n & 1) ? pingpong : res;
    }

    ////////////////////// Forward FFT //////////////////////

    // template<std::size_t N>
    // std::array<vec2, N> fft(const std::array<float, N>& p)
    // {
    //     constexpr auto log2_n = static_cast<std::size_t>(std::log2(N));
    //     // 1. Since this isn't an inlined version of the algorithm
    //     // we take O(N) space complexity with the Bit-reverse copy step
    //     std::array<vec2, N> br;
    //     for(std::size_t i = 0 ; i < N ; ++i)
    //     {
    //         br[bit_reverse(i, log2_n)] = vec2{p[i], 0};
    //     }

    //     for(std::size_t s = 1 ; s <= log2_n ; ++s)
    //     {
    //         // TODO: change pow and log for bit shifting operations
    //         auto m = static_cast<std::size_t>(std::pow(2, s));
    //         const auto wm = euler(-2 * pi / m);

    //         for(std::size_t k = 0; k < N; k += m)
    //         {
    //             auto w = vec2{1,0};

    //             const auto half_m = m/2;
    //             for(std::size_t j = 0; j < half_m; ++j)
    //             {
    //                 const auto t = complex_mul(w, br[k + j + half_m]);
    //                 auto u = br[k + j];
    //                 br[k + j] = complex_add(u, t);
    //                 br[k + j + half_m] = complex_sub(u, t);
    //                 w = complex_mul(w, wm);
    //             }
    //         }
    //     }

    //     return br;
    // }

    ////////////////////// Inverse FFT //////////////////////

    // template<std::size_t N>
    // std::array<vec2, N> ifft(const std::array<vec2, N>& p)
    // {

    //     constexpr auto log2_n = static_cast<std::size_t>(std::log2(N));
    //     // 1. Since this isn't an inlined version of the algorithm
    //     // we take O(N) space complexity with the Bit-reverse copy step
    //     std::array<vec2, N> br;
    //     for(std::size_t i = 0 ; i < N ; ++i)
    //     {
    //         br[bit_reverse(i, log2_n)] = p[i];
    //     }

    //     for(std::size_t s = 1 ; s <= log2_n ; ++s)
    //     {
    //         auto m = static_cast<std::size_t>(std::pow(2, s));
    //         const auto wm = euler(2 * pi / m);

    //         for(std::size_t k = 0; k < N; k += m)
    //         {
    //             auto w = vec2{1,0};

    //             const auto half_m = m/2;
    //             for(std::size_t j = 0; j < half_m; ++j)
    //             {
    //                 const auto t = complex_mul(w, br[k + j + half_m]);
    //                 auto u = br[k + j];
    //                 br[k + j] = complex_add(u, t);
    //                 br[k + j + half_m] = complex_sub(u, t);
    //                 w = complex_mul(w, wm);
    //             }
    //         }
    //     }

    //     // Devide result by N
    //     // Structured bindings for real and imaginary part
    //     for(auto& [r, i] : br)
    //     {
    //         // Only real part needs the division
    //         // since the imaginary component will be 0
    //         r /= N;
    //     }

    //     return br;
    // }

}

/////////////////////////////////////////////////////////

#endif // FFT_DIT_PARALLEL_HPP