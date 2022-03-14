#ifndef FFT_DIT_ITER_HPP
#define FFT_DIT_ITER_HPP

// #include <type_traits>
// #include <concepts>
#include <cstdint>
#include "utils.hpp"

namespace dit
{
    ////////////////////// Forward FFT //////////////////////

    template<std::size_t N>
    std::array<vec2, N> fft(const std::array<float, N>& p)
    {
        constexpr auto log2_n = static_cast<std::size_t>(std::log2(N));
        // 1. Since this isn't an inlined version of the algorithm
        // we take O(N) space complexity with the Bit-reverse copy step
        std::array<vec2, N> br;
        for(std::size_t i = 0 ; i < N ; ++i)
        {
            br[bit_reverse(i, log2_n)] = vec2{p[i], 0};
        }

        for(std::size_t s = 1 ; s <= log2_n ; ++s)
        {
            // TODO: change pow and log for bit shifting operations
            auto m = static_cast<std::size_t>(std::pow(2, s));
            const auto wm = euler(-2 * pi / m);

            for(std::size_t k = 0; k < N; k += m)
            {
                auto w = vec2{1,0};

                const auto half_m = m/2;
                for(std::size_t j = 0; j < half_m; ++j)
                {
                    const auto t = complex_mul(w, br[k + j + half_m]);
                    auto u = br[k + j];
                    br[k + j] = complex_add(u, t);
                    br[k + j + half_m] = complex_sub(u, t);
                    w = complex_mul(w, wm);
                }
            }
        }

        return br;
    }

    ////////////////////// Inverse FFT //////////////////////

    template<std::size_t N>
    std::array<vec2, N> ifft(const std::array<vec2, N>& p)
    {

        constexpr auto log2_n = static_cast<std::size_t>(std::log2(N));
        // 1. Since this isn't an inlined version of the algorithm
        // we take O(N) space complexity with the Bit-reverse copy step
        std::array<vec2, N> br;
        for(std::size_t i = 0 ; i < N ; ++i)
        {
            br[bit_reverse(i, log2_n)] = p[i];
        }

        for(std::size_t s = 1 ; s <= log2_n ; ++s)
        {
            auto m = static_cast<std::size_t>(std::pow(2, s));
            const auto wm = euler(2 * pi / m);

            for(std::size_t k = 0; k < N; k += m)
            {
                auto w = vec2{1,0};

                const auto half_m = m/2;
                for(std::size_t j = 0; j < half_m; ++j)
                {
                    const auto t = complex_mul(w, br[k + j + half_m]);
                    auto u = br[k + j];
                    br[k + j] = complex_add(u, t);
                    br[k + j + half_m] = complex_sub(u, t);
                    w = complex_mul(w, wm);
                }
            }
        }

        // Devide result by N
        // Structured bindings for real and imaginary part
        for(auto& [r, i] : br)
        {
            // Only real part needs the division
            // since the imaginary component will be 0
            r /= N;
        }

        return br;
    }

}

/////////////////////////////////////////////////////////

#endif // FFT_DIT_ITER_HPP