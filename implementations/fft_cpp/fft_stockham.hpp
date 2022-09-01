#ifndef FFT_STOCKHAM_HPP
#define FFT_STOCKHAM_HPP

#include <cstdint>
#include "utils.hpp"


namespace stockham
{
    ////////////////////// Forward FFT //////////////////////

    template<std::size_t N>
    std::array<vec2, N> fft(const std::array<vec2, N>& tmp)
    {
        std::array<vec2, N> p[2];

        p[0] = tmp;
        constexpr auto log2_n = static_cast<std::size_t>(std::log2(N));

        // TODO: change log_2 to shifting iteration
        // (increment s while (1 << s) < N)
        // NOTE: ITERATES OVER ALL STAGES
        // FIXME: this loop might need to start in 0
        for(std::size_t stage = 0 ; s < log2_n ; ++stage)
        {
            const auto group_size = N >> stage;
            const auto half_gs = group_size >> 1;
            const auto s = 1 << stage;

            const auto wm = euler(-2 * pi / group_size);

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
                    auto a = p[s % 2][k + s*(j + 0)];
                    auto b = p[s % 2][k + s*(j + half_gs)];
                    // A + B
                    p[(s+1) % 2][k + s*(2*j + 0)] = complex_add(a, b);
                    // (A - B) * W
                    p[(s+1) % 2][k + s*(2*j + 1)] = complex_mul(complex_sub(a, b), w);
                    w = complex_mul(w, wm);
                }
            }
        }

        return p[log2_n % 2];
    }

    ////////////////////// Inverse FFT //////////////////////

    template<std::size_t N>
    std::array<vec2, N> ifft(const std::array<vec2, N>& tmp)
    {
        std::array<vec2, N> p[2];

        p[0] = tmp;
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
                    auto a = p[s % 2][k + (1 << s)*(j + 0)];
                    auto b = p[s % 2][k + (1 << s)*(j + half_gs)];
                    // A + B
                    p[(s+1) % 2][k + (1 << s)*(2*j + 0)] = complex_add(a, b);
                    // (A - B) * W
                    p[(s+1) % 2][k + (1 << s)*(2*j + 1)] = complex_mul(complex_sub(a, b), w);
                    w = complex_mul(w, wm);
                }
            }
        }

        return p[log2_n % 2];
    }

}

/////////////////////////////////////////////////////////

#endif // FFT_STOCKHAM_HPP
