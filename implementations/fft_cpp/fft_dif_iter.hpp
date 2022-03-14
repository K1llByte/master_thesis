#ifndef FFT_DIF_ITER_HPP
#define FFT_DIF_ITER_HPP

// #include <type_traits>
// #include <concepts>
#include <cstdint>
#include "utils.hpp"

namespace dif
{
    ////////////////////// Forward FFT //////////////////////

    template<std::size_t N>
    std::array<vec2, N> fft(const std::array<vec2, N>& tmp)
    {
        auto p = tmp;
        constexpr auto log2_n = static_cast<std::size_t>(std::log2(N));

        // TODO: change log_2 to shifting iteration
        // (increment s while (1 << s) < N)
        // NOTE: ITERATES OVER ALL STAGES
        // FIXME: this loop might need to start in 0

        for(std::size_t s = 0 ; s < log2_n ; ++s)
        {
            const auto group_size = N >> s;
            const auto half_gs = group_size >> 1;

            const auto wm = euler(-2 * pi / group_size);

            // NOTE: ITERATES OVER ALL GROUPS
            for(std::size_t k = 0; k < N; k += group_size)
            {
                auto w = vec2{1,0};

                // NOTE: COMPUTES BASIC BUTTERFLIES
                for(std::size_t j = 0; j < half_gs; ++j)
                {
                    // The DIF basic butterfly operation computes
                    // two values
                    const auto b = p[k + j + half_gs];
                    const auto a = p[k + j];
                    // A + B
                    p[k + j] = complex_add(a, b);
                    // (A - B) * W
                    p[k + j + half_gs] = complex_mul(complex_sub(a, b), w);
                    w = complex_mul(w, wm);
                }
            }
        }

        // Since this isn't an inlined version of the algorithm
        // we take O(N) space complexity with the Bit-reverse copy step
        std::array<vec2, N> br;
        for(std::size_t i = 0 ; i < N ; ++i)
        {
            br[bit_reverse(i, log2_n)] = p[i];
        }

        return br;
    }

    ////////////////////// Inverse FFT //////////////////////

    template<std::size_t N>
    std::array<vec2, N> ifft(const std::array<vec2, N>& tmp)
    {
        auto p = tmp;
        constexpr auto log2_n = static_cast<std::size_t>(std::log2(N));

        // TODO: change log_2 to shifting iteration
        // (increment s while (1 << s) < N)
        // NOTE: ITERATES OVER ALL STAGES
        // FIXME: this loop might need to start in 0
        for(std::size_t s = 0 ; s < log2_n ; ++s)
        {
            const auto group_size = N >> s;
            const auto half_gs = group_size >> 1;

            const auto wm = euler(2 * pi / group_size);

            // NOTE: ITERATES OVER ALL GROUPS
            for(std::size_t k = 0; k < N; k += group_size)
            {
                auto w = vec2{1,0};
                // const auto half_gs = group_size >> 1;

                // NOTE: COMPUTES BASIC BUTTERFLIES
                for(std::size_t j = 0; j < half_gs; ++j)
                {
                    // The DIF basic butterfly operation computes
                    // two values
                    auto b = p[k + j + half_gs];
                    auto a = p[k + j];
                    // A + B
                    p[k + j] = complex_add(a, b);
                    // (A - B) * W
                    p[k + j + half_gs] = complex_mul(complex_sub(a, b), w);
                    w = complex_mul(w, wm);
                }
            }
        }

        // Since this isn't an inlined version of the algorithm
        // we take O(N) space complexity with the Bit-reverse copy step
        // Devide result by N
        std::array<vec2, N> br;
        for(std::size_t i = 0 ; i < N ; ++i)
        {
            br[bit_reverse(i, log2_n)] = vec2{p[i][0] / N, p[i][1]};
        }

        return br;
    }

}

/////////////////////////////////////////////////////////

#endif // FFT_DIF_ITER_HPP