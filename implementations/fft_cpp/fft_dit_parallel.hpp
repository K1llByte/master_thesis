#ifndef FFT_DIT_PARALLEL_HPP
#define FFT_DIT_PARALLEL_HPP

// #include <type_traits>
// #include <concepts>
#include <cstdint>
#include <vector>
#include <thread>
#include <barrier>
#include <sstream>

#include "utils.hpp"

namespace dit::parallel
{
    //////////////// Forward FFT Entry Point ////////////////

    template<std::size_t N>
    const std::array<vec2, N> fft(const std::array<vec2, N>& p)
    {
        // Since this isn't an inlined version of the algorithm
        // we take O(N) space complexity with the Bit-reverse copy step
        std::array<vec2, N> res = p;
        std::array<vec2, N> pingpong;
        inplace_bit_reversal(res);

        constexpr auto log_n = static_cast<std::size_t>(std::log2(N));
        constexpr auto num_processors = log_n;
        std::barrier sync_point(num_processors);

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

                    const std::size_t group_size = 2 << stage;
                    for(std::size_t val = 0; val < log_n; ++val)
                    {
                        const auto idx = id*log_n + val;
                        // const auto w = euler(-2 * pi / group_size * (idx % group_size));
                        const auto half_group_size = group_size >> 1;
                        if((idx % group_size) >= half_group_size)
                        {
                            const auto w = euler(-2 * pi / group_size * ((idx % group_size) % half_group_size));
                            (*to)[idx] = complex_sub((*from)[idx-half_group_size], complex_mul(w, (*from)[idx]));
                        }
                        else
                        {
                            const auto w = euler(-2 * pi / group_size * ((idx % group_size) % half_group_size));
                            (*to)[idx] = complex_add((*from)[idx], complex_mul(w, (*from)[idx+half_group_size]));
                        }
                    }
                    sync_point.arrive_and_wait();
                }
            }, i);
        }

        for(auto& processor : processors)
            processor.join();
        
        return (log_n & 1) ? pingpong : res;
    }

    //////////////// Inverse FFT Entry Point ////////////////

    template<std::size_t N>
    const std::array<vec2, N> ifft(const std::array<vec2, N>& p)
    {
        // // Since this isn't an inlined version of the algorithm
        // // we take O(N) space complexity with the Bit-reverse copy step
        // std::array<vec2, N> br;
        // for(std::size_t i = 0 ; i < N ; ++i)
        // {
        //     br[bit_reverse(i, log2_n)] = vec2{p[i], 0};
        // }
        std::array<vec2, N> res = p;
        std::array<vec2, N> pingpong;
        inplace_bit_reversal(res);

        constexpr auto log_n = static_cast<std::size_t>(std::log2(N));
        constexpr auto num_processors = log_n;
        std::barrier sync_point(num_processors);

        std::vector<std::thread> processors(num_processors);
        for(std::size_t i = 0; i < num_processors; ++i)
        {
            // Concurrent processors
            processors[i] = std::thread([&](std::size_t id)
            {
                for(std::size_t stage = 0; stage < log_n; ++stage)
                {
                    // TODO: 2 conditions check not optimal, change this 
                    auto& from = (stage % 2 == 0) ? res : pingpong;
                    auto& to = (stage % 2 == 0) ? pingpong : res;
                    // Since there are log2(N) processors each
                    // one is gonna compute 2 fft dit butterflies

                    // Temporary storage to avoid read access
                    // race conditions
                    // TODO: This array might not be needed if the
                    // barrier waits after write
                    // std::array<vec2, log_n> tmp;

                    const auto group_size = 2 << stage;
                    for(std::size_t val = 0; val < log_n; ++val)
                    {
                        const auto idx = id*log_n + val;
                        const auto w = euler(2 * pi / group_size * (idx % group_size));
                        const auto half_group_size = group_size >> 1;
                        if((idx % group_size) >= half_group_size)
                        {
                            to[idx] = complex_add(from[idx], complex_mul(w, from[idx-half_group_size]));
                        }
                        else
                        {
                            to[idx] = complex_sub(res[idx+half_group_size], complex_mul(w, from[idx]));
                        }
                    }

                    // Sync all processors stage
                    std::stringstream ss;
                    ss << "ID: " << id << ", Stage: " << stage << " waiting ...\n";
                    std::cout << ss.str();
                    sync_point.arrive_and_wait();
                }
            }, i);
        }

        for(auto& processor : processors)
            processor.join();
        
        return res;
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